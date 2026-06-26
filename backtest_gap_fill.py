#!/usr/bin/env python3
"""
backtest_gap_fill.py — Morning Gap Fill Strategy (SPY / QQQ / IWM, 5-min bars)

Signal rules (mirror of app.py score_gap_fill):
  - Gap of 0.3–0.5% between prior day close and today's open
  - Long if gap down, Short if gap up (fading the gap back to prior close)
  - Opening drive filter: first 5-min bar must close AGAINST the gap
    (gap down → first bar closes UP; gap up → first bar closes DOWN)
  - Entry at today's open price
  - Target: prior close (full fill)
  - Stop: 1.5× the gap size (away from entry)
  - Signal expires at 11:00 AM if not yet triggered

Win conditions measured:
  1. Target hit (price reached prior close) before stop or 11am cutoff
  2. Direction accuracy at +15m / +30m / +60m from entry bar

Run:  python backtest_gap_fill.py
Output: backtest_results/gap_fill_report.html
"""

import os, sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
from dotenv import load_dotenv

load_dotenv()

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except ImportError:
    sys.exit("alpaca-py not installed.")

ET         = pytz.timezone("US/Eastern")
API_KEY    = os.environ.get("ALPACA_API_KEY", "")
API_SECRET = os.environ.get("ALPACA_SECRET_KEY", "")

SYMBOLS     = ["SPY", "QQQ", "IWM"]
LOOKBACK    = 365
GAP_MIN     = 0.003   # 0.3%
GAP_MAX     = 0.005   # 0.5%
STOP_MULT   = 1.5     # stop = 1.5× gap size
WINDOWS     = [15, 30, 60]
MIN_SAMPLES = 5

CACHE_DIR   = Path("backtest_cache")
RESULTS_DIR = Path("backtest_results")
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────

def _normalize(df, symbol):
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        ts_col = next((c for c in df.columns if "time" in c.lower()), None)
        if ts_col:
            df = df.set_index(ts_col)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    return df


def fetch_bars(symbol, timeframe, days, label):
    cache = CACHE_DIR / f"{symbol}_{label}_{days}d.pkl"
    if cache.exists():
        age_h = (datetime.now().timestamp() - cache.stat().st_mtime) / 3600
        if age_h < 12:
            print(f"  using cached {symbol} {label}")
            return pd.read_pickle(cache)
    print(f"  fetching {symbol} {label} ({days}d)…")
    client = StockHistoricalDataClient(API_KEY, API_SECRET)
    now = datetime.now(ET)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=now - timedelta(days=days),
        end=now,
        feed="iex",
    )
    df = _normalize(client.get_stock_bars(req).df, symbol)
    df.to_pickle(cache)
    return df


# ─────────────────────────────────────────────
# Signal detection
# ─────────────────────────────────────────────

def detect_signals(df_5m, df_1d, symbol):
    """
    For each trading day, check if a gap fill signal fired.
    Returns list of result dicts, one per qualifying day.
    """
    rows = []

    # Build prior-close lookup from daily bars
    daily = df_1d.copy()
    daily.index = pd.to_datetime(daily.index)
    daily_dates = sorted(set(daily.index.date))

    for date in sorted(set(df_5m.index.date)):
        day_5m = df_5m[df_5m.index.date == date].sort_index()
        if len(day_5m) < 2:
            continue

        # Prior close from daily bars
        prior_days = [d for d in daily_dates if d < date]
        if not prior_days:
            continue
        prior_close = float(daily[daily.index.date == prior_days[-1]]["close"].iloc[-1])

        bar0       = day_5m.iloc[0]
        today_open = float(bar0["open"])
        gap        = today_open - prior_close
        gap_pct    = abs(gap) / prior_close
        direction  = "short" if gap > 0 else "long"

        # Must be in range
        if gap_pct < GAP_MIN or gap_pct > GAP_MAX:
            continue

        # Opening drive filter: first bar must close against the gap
        bar0_close = float(bar0["close"])
        bar0_open  = float(bar0["open"])
        if direction == "long"  and bar0_close < bar0_open:
            continue   # gap down but first bar also closed down → skip
        if direction == "short" and bar0_close > bar0_open:
            continue   # gap up but first bar also closed up → skip

        entry  = today_open
        target = prior_close
        stop   = entry + STOP_MULT * abs(gap) if direction == "short" else entry - STOP_MULT * abs(gap)

        # --- Outcome 1: did price hit target before stop (before 11am)? ---
        cutoff_bars = day_5m[
            (day_5m.index.hour < 11) |
            ((day_5m.index.hour == 11) & (day_5m.index.minute == 0))
        ]
        target_hit = False
        stop_hit   = False
        for _, bar in cutoff_bars.iterrows():
            hi, lo = float(bar["high"]), float(bar["low"])
            if direction == "long":
                if hi >= target:
                    target_hit = True; break
                if lo <= stop:
                    stop_hit = True; break
            else:
                if lo <= target:
                    target_hit = True; break
                if hi >= stop:
                    stop_hit = True; break

        # --- Outcome 2: direction accuracy at +15/+30/+60m ---
        # Entry is at bar index 1 (after the opening drive filter bar)
        entry_idx = 1
        rets = {}
        for w in WINDOWS:
            bars_fwd = w // 5
            fwd_idx  = entry_idx + bars_fwd
            if fwd_idx < len(day_5m):
                fwd_close = float(day_5m.iloc[fwd_idx]["close"])
                ret = (fwd_close - entry) / entry * 100
                rets[w] = ret * (1 if direction == "long" else -1)
            else:
                rets[w] = np.nan

        rows.append({
            "date":       date,
            "symbol":     symbol,
            "direction":  direction,
            "gap_pct":    round(gap_pct * 100, 3),
            "entry":      round(entry, 2),
            "target":     round(target, 2),
            "stop":       round(stop, 2),
            "target_hit": target_hit,
            "stop_hit":   stop_hit,
            "ret_15":     rets[15],
            "ret_30":     rets[30],
            "ret_60":     rets[60],
            "up_15":      rets[15] > 0 if not pd.isna(rets[15]) else np.nan,
            "up_30":      rets[30] > 0 if not pd.isna(rets[30]) else np.nan,
            "up_60":      rets[60] > 0 if not pd.isna(rets[60]) else np.nan,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────

CSS = """
body{background:#0b0b0f;color:#d1d5db;font-family:'Segoe UI',system-ui,sans-serif;font-size:13px;padding:28px;max-width:1000px;margin:0 auto}
h1{font-size:22px;font-weight:700;color:#f9fafb;margin:0 0 4px}
h2{font-size:12px;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:.1em;margin:32px 0 12px;border-bottom:1px solid #1f2028;padding-bottom:8px}
h3{font-size:13px;font-weight:600;color:#9ca3af;margin:20px 0 6px}
p.sub{color:#6b7280;font-size:12px;margin:0 0 16px;line-height:1.6}
.meta{color:#4b5563;font-size:12px;margin-bottom:28px}
.card{background:#111118;border:1px solid #1f2028;border-radius:10px;overflow:hidden;margin-bottom:16px}
table{border-collapse:collapse;width:100%}
th{text-align:left;padding:8px 12px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:#4b5563;border-bottom:1px solid #1f2028;background:#0f0f18;white-space:nowrap}
td{padding:7px 12px;border-bottom:1px solid #141420;font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:none}
tr:hover td{background:#13131e}
.g{color:#4ade80;font-weight:600}.y{color:#facc15}.n{color:#6b7280}.r{color:#f87171;font-weight:600}
.sym{font-weight:700;color:#f9fafb}
.disc{background:#0e0e18;border:1px solid #1f2028;border-radius:8px;padding:14px 16px;color:#4b5563;font-size:12px;line-height:1.7;margin-top:36px}
.disc strong{color:#6b7280}
"""


def _pct(v):
    if pd.isna(v): return '<td class="n">—</td>'
    p   = v * 100
    cls = "g" if p >= 60 else ("y" if p >= 55 else ("n" if p >= 48 else "r"))
    return f'<td class="{cls}">{p:.1f}%</td>'


def _ret(v):
    if pd.isna(v): return '<td class="n">—</td>'
    cls = "g" if v > 0.05 else ("y" if v > 0 else "r")
    return f'<td class="{cls}">{v:+.2f}%</td>'


def summary_table(sigs, title, sub=""):
    rows = ""
    for sym in (["All"] + SYMBOLS):
        sub_df = sigs if sym == "All" else sigs[sigs["symbol"] == sym]
        n = len(sub_df)
        if n < MIN_SAMPLES:
            continue
        wr_target = sub_df["target_hit"].mean()
        wr_15 = sub_df["up_15"].dropna().mean()
        wr_30 = sub_df["up_30"].dropna().mean()
        wr_60 = sub_df["up_60"].dropna().mean()
        avg_15 = sub_df["ret_15"].dropna().mean()
        avg_30 = sub_df["ret_30"].dropna().mean()
        avg_60 = sub_df["ret_60"].dropna().mean()
        long_n  = int((sub_df["direction"] == "long").sum())
        short_n = int((sub_df["direction"] == "short").sum())
        rows += (
            f'<tr>'
            f'<td class="sym">{sym}</td>'
            f'<td>{n} ({long_n}L/{short_n}S)</td>'
            f'{_pct(wr_target)}'
            f'{_pct(wr_15)}{_pct(wr_30)}{_pct(wr_60)}'
            f'{_ret(avg_15)}{_ret(avg_30)}{_ret(avg_60)}'
            f'</tr>\n'
        )
    sub_html = f"<p class='sub'>{sub}</p>" if sub else ""
    return f"""<h3>{title}</h3>{sub_html}<div class="card"><table>
<tr>
  <th>Symbol</th><th>Signals</th>
  <th>Target Hit%</th>
  <th>Win% +15m</th><th>Win% +30m</th><th>Win% +60m</th>
  <th>Avg +15m</th><th>Avg +30m</th><th>Avg +60m</th>
</tr>
{rows}</table></div>"""


def direction_table(sigs, title):
    rows = ""
    for direction, label in [("long", "↑ Long (gap down, Buy Calls)"), ("short", "↓ Short (gap up, Buy Puts)")]:
        sub = sigs[sigs["direction"] == direction]
        n   = len(sub)
        if n < MIN_SAMPLES:
            continue
        wr_target = sub["target_hit"].mean()
        wr_15 = sub["up_15"].dropna().mean()
        wr_30 = sub["up_30"].dropna().mean()
        wr_60 = sub["up_60"].dropna().mean()
        avg_15 = sub["ret_15"].dropna().mean()
        avg_30 = sub["ret_30"].dropna().mean()
        avg_60 = sub["ret_60"].dropna().mean()
        rows += (
            f'<tr>'
            f'<td class="sym">{label}</td>'
            f'<td>{n}</td>'
            f'{_pct(wr_target)}'
            f'{_pct(wr_15)}{_pct(wr_30)}{_pct(wr_60)}'
            f'{_ret(avg_15)}{_ret(avg_30)}{_ret(avg_60)}'
            f'</tr>\n'
        )
    return f"""<h3>{title}</h3><div class="card"><table>
<tr>
  <th>Direction</th><th>Signals</th>
  <th>Target Hit%</th>
  <th>Win% +15m</th><th>Win% +30m</th><th>Win% +60m</th>
  <th>Avg +15m</th><th>Avg +30m</th><th>Avg +60m</th>
</tr>
{rows}</table></div>"""


def gap_size_table(sigs, title):
    """Break down by gap size bucket."""
    sigs = sigs.copy()
    sigs["gap_bucket"] = pd.cut(sigs["gap_pct"], bins=[0.3, 0.35, 0.4, 0.45, 0.5], labels=["0.30–0.35%","0.35–0.40%","0.40–0.45%","0.45–0.50%"])
    rows = ""
    for bucket, grp in sigs.groupby("gap_bucket", observed=True):
        n = len(grp)
        if n < MIN_SAMPLES:
            continue
        rows += (
            f'<tr>'
            f'<td class="sym">{bucket}</td>'
            f'<td>{n}</td>'
            f'{_pct(grp["target_hit"].mean())}'
            f'{_pct(grp["up_15"].dropna().mean())}'
            f'{_pct(grp["up_30"].dropna().mean())}'
            f'{_pct(grp["up_60"].dropna().mean())}'
            f'</tr>\n'
        )
    if not rows:
        return ""
    return f"""<h3>{title}</h3><div class="card"><table>
<tr><th>Gap Size</th><th>Signals</th><th>Target Hit%</th><th>Win% +15m</th><th>Win% +30m</th><th>Win% +60m</th></tr>
{rows}</table></div>"""


def recent_table(sigs, n=30):
    rows = ""
    for _, r in sigs.tail(n).iterrows():
        d   = r["direction"]
        cls = "g" if d == "long" else "r"
        tgt = "✅" if r["target_hit"] else ("🛑" if r["stop_hit"] else "—")
        rows += (
            f'<tr>'
            f'<td class="sym">{r["date"]}</td>'
            f'<td>{r["symbol"]}</td>'
            f'<td class="{cls}">{"↑ Long" if d=="long" else "↓ Short"}</td>'
            f'<td>{r["gap_pct"]:.2f}%</td>'
            f'<td>${r["entry"]:.2f}</td>'
            f'<td>${r["target"]:.2f}</td>'
            f'<td>${r["stop"]:.2f}</td>'
            f'<td style="text-align:center">{tgt}</td>'
            f'{_ret(r["ret_15"])}{_ret(r["ret_30"])}{_ret(r["ret_60"])}'
            f'</tr>\n'
        )
    return f"""<h3>Most Recent {n} Signals</h3><div class="card"><table>
<tr>
  <th>Date</th><th>Symbol</th><th>Side</th><th>Gap</th>
  <th>Entry</th><th>Target</th><th>Stop</th><th>Fill?</th>
  <th>Ret +15m</th><th>Ret +30m</th><th>Ret +60m</th>
</tr>
{rows}</table></div>"""


def build_report(all_sigs):
    now_str = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    n_total = len(all_sigs)
    body    = ""

    body += "<h2>Overall Results</h2>"
    body += summary_table(all_sigs, "All symbols combined",
        sub="Target Hit% = price reached prior close before stop or 11am. Win% = price moved in trade direction at each hold window.")
    body += direction_table(all_sigs, "By Direction (all symbols)")
    body += "<h2>By Gap Size</h2>"
    body += gap_size_table(all_sigs, "Win rate by gap size bucket (all symbols)")
    body += "<h2>Recent Signal Log</h2>"
    body += recent_table(all_sigs, n=40)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Gap Fill Backtest</title>
<style>{CSS}</style></head><body>
<h1>Morning Gap Fill Backtest — SPY / QQQ / IWM</h1>
<div class="meta">
  Generated {now_str} &nbsp;·&nbsp; {LOOKBACK}-day lookback &nbsp;·&nbsp;
  {n_total} total signals &nbsp;·&nbsp; Gap range: {GAP_MIN*100:.1f}–{GAP_MAX*100:.1f}% &nbsp;·&nbsp;
  Stop: {STOP_MULT}× gap &nbsp;·&nbsp; Cutoff: 11:00 AM ET
</div>
<p class="sub">
  <strong style="color:#9ca3af">Signal rules:</strong>
  Gap of 0.3–0.5% at open vs prior close ·
  Opening drive filter (first 5-min bar must close against the gap direction) ·
  Entry at open · Target = prior close (full fill) · Stop = {STOP_MULT}× gap size ·
  Expires 11am ET
</p>
{body}
<div class="disc">
  <strong>Target Hit%</strong> = price reached the prior close (full gap fill) before hitting the stop or the 11am cutoff.
  <strong>Win%</strong> = price moved in the trade's direction at the measured window — 50% is a coin flip.
  <span style="color:#4ade80">≥60% = meaningful edge</span>,
  <span style="color:#facc15">55–60% = slight edge</span>,
  gray = noise,
  <span style="color:#f87171">&lt;48% = works against you</span>.
  Min {MIN_SAMPLES} signals per row. Past results don't predict the future.
</div>
</body></html>"""


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    if not API_KEY or not API_SECRET:
        sys.exit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")

    print(f"Gap Fill Backtest — {', '.join(SYMBOLS)} — {LOOKBACK} days")
    print("=" * 60)

    all_sigs = []
    for sym in SYMBOLS:
        print(f"\n{sym}:")
        df_5m = fetch_bars(sym, TimeFrame(5, TimeFrameUnit.Minute), LOOKBACK, "5m")
        df_1d = fetch_bars(sym, TimeFrame(1, TimeFrameUnit.Day),    LOOKBACK, "1d")

        df_5m = df_5m[
            ((df_5m.index.hour == 9)  & (df_5m.index.minute >= 30)) |
            ((df_5m.index.hour >= 10) & (df_5m.index.hour < 16))
        ].copy()

        sigs = detect_signals(df_5m, df_1d, sym)
        if sigs.empty:
            print(f"  no signals found")
            continue
        print(f"  {len(sigs)} signals  ({int((sigs['direction']=='long').sum())}L / {int((sigs['direction']=='short').sum())}S)")
        print(f"  target hit rate: {sigs['target_hit'].mean()*100:.1f}%")
        for w in WINDOWS:
            wr = sigs[f"up_{w}"].dropna().mean()
            ar = sigs[f"ret_{w}"].dropna().mean()
            print(f"  +{w:2d}m  win={wr*100:.1f}%  avg={ar:+.2f}%")
        all_sigs.append(sigs)

    if not all_sigs:
        print("No signals found across any symbol.")
        return

    combined = pd.concat(all_sigs, ignore_index=True)
    print(f"\nTotal: {len(combined)} signals across all symbols")

    out = RESULTS_DIR / "gap_fill_report.html"
    out.write_text(build_report(combined), encoding="utf-8")
    print(f"\nReport → {out.resolve()}")


if __name__ == "__main__":
    main()
