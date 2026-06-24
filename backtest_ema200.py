#!/usr/bin/env python3
"""
backtest_ema200.py — EMA 200 Pullback Bounce Strategy (SPY, 5-minute bars)

Signal definition
-----------------
Long (Calls):
  1. EMA 200 is rising (EMA[i] > EMA[i - SLOPE_BARS])
  2. Previous bar closed ABOVE EMA 200 (trend intact before pullback)
  3. Current bar's LOW touches or dips below EMA 200 (the pullback)
  4. Current bar closes ABOVE EMA 200 (the bounce confirmed)
  5. Current bar is green (close > open)

Short (Puts): exact mirror — falling EMA 200, prior bar below, high
  touches EMA from below, bar closes below EMA, red candle.

Outcomes measured: did SPY close in the trade's direction 15 / 30 / 60 min later?

Run:  python backtest_ema200.py
Output: backtest_results/ema200_report.html
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
    sys.exit("alpaca-py not installed. Run: pip install alpaca-py")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

ET         = pytz.timezone("US/Eastern")
API_KEY    = os.environ.get("ALPACA_API_KEY", "")
API_SECRET = os.environ.get("ALPACA_SECRET_KEY", "")

SYMBOL      = "SPY"
LOOKBACK    = 365          # calendar days of history
EMA_PERIOD  = 200          # the key EMA
SLOPE_BARS  = 5            # bars used to determine EMA slope direction
WINDOWS     = [15, 30, 60] # forward measurement windows in minutes
MIN_SAMPLES = 5            # suppress rows with too few signals
PROXIMITY   = 0.002        # bar's low/high must be within 0.2% of EMA to count as "touch"

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
            print(f"  using cached {symbol} {label} ({days}d)")
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
# Indicators & outcomes
# ─────────────────────────────────────────────

def add_indicators(df):
    df = df.copy()
    df["ema200"]  = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    by_day        = df.groupby(df.index.date)
    df["rel_vol"] = df["volume"] / by_day["volume"].transform("mean")
    # distance of close from EMA200, as percent
    df["ema_dist_pct"] = (df["close"] - df["ema200"]) / df["ema200"] * 100
    return df


def add_outcomes(df):
    df   = df.copy()
    bpw  = {15: 3, 30: 6, 60: 12}   # bars per window on 5-min chart
    for w, bars in bpw.items():
        fwd = df.groupby(df.index.date)["close"].transform(lambda x: x.shift(-bars))
        df[f"ret_{w}"]  = (fwd - df["close"]) / df["close"] * 100
        df[f"up_{w}"]   = df[f"ret_{w}"] > 0
    return df


# ─────────────────────────────────────────────
# Signal detection
# ─────────────────────────────────────────────

def detect_signals(df):
    """
    Scan every bar for a long or short EMA-200 pullback bounce.
    Returns a DataFrame with one row per signal.
    """
    rows   = []
    close  = df["close"].values
    open_  = df["open"].values
    low    = df["low"].values
    high   = df["high"].values
    ema    = df["ema200"].values
    dates  = np.array(df.index.date)
    idx    = df.index

    # track last signal index to avoid flagging the same bounce twice in a row
    last_long_i  = -99
    last_short_i = -99

    for i in range(SLOPE_BARS + 1, len(df)):
        # don't look across day boundaries for the slope
        if dates[i] != dates[i - 1]:
            last_long_i  = -99
            last_short_i = -99

        e     = ema[i]
        e_ago = ema[i - SLOPE_BARS]
        c, o, lo, hi = close[i], open_[i], low[i], high[i]
        pc    = close[i - 1]

        ema_rising  = e > e_ago
        ema_falling = e < e_ago

        # Long: EMA rising, prior bar above EMA, this bar touches EMA from above and bounces
        long_touch  = lo <= e * (1 + PROXIMITY)  # low dipped to/below EMA (with tolerance)
        long_sig = (
            ema_rising
            and pc > ema[i - 1]          # prior bar was above EMA (uptrend intact)
            and long_touch               # current bar tagged the EMA
            and c > e                    # closes back above EMA
            and c > o                    # green candle (momentum confirmation)
            and i - last_long_i > 1      # not a repeat of the immediately prior bar
        )

        # Short: EMA falling, prior bar below EMA, this bar touches EMA from below and reverses
        short_touch = hi >= e * (1 - PROXIMITY)
        short_sig = (
            ema_falling
            and pc < ema[i - 1]
            and short_touch
            and c < e                    # closes back below EMA
            and c < o                    # red candle
            and i - last_short_i > 1
        )

        if long_sig:
            rows.append(_make_row(idx[i], df.iloc[i], "long"))
            last_long_i = i
        elif short_sig:
            rows.append(_make_row(idx[i], df.iloc[i], "short"))
            last_short_i = i

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("timestamp")


def _make_row(ts, bar, direction):
    r = {
        "timestamp":   ts,
        "direction":   direction,
        "price":       float(bar["close"]),
        "ema200":      float(bar["ema200"]),
        "ema_dist_pct": float(bar["ema_dist_pct"]),
        "rel_vol":     float(bar["rel_vol"]),
        "hour":        ts.hour,
        "time_slot":   f"{ts.hour:02d}:{str(ts.minute).zfill(2)}",
        "hour_slot":   f"{ts.hour:02d}:00",
    }
    for w in WINDOWS:
        r[f"ret_{w}"] = bar[f"ret_{w}"]
        r[f"up_{w}"]  = bar[f"up_{w}"]
    return r


# ─────────────────────────────────────────────
# Accuracy helpers
# ─────────────────────────────────────────────

def winrate(sub, direction, window):
    col = f"up_{window}"
    s   = sub[sub["direction"] == direction][col].dropna()
    n   = len(s)
    if n < MIN_SAMPLES:
        return np.nan, n
    acc = s.mean() if direction == "long" else (1 - s.mean())
    return float(acc), n


def avg_ret(sub, direction, window):
    col = f"ret_{window}"
    s   = sub[sub["direction"] == direction][col].dropna()
    if len(s) < MIN_SAMPLES:
        return np.nan
    sign = 1 if direction == "long" else -1
    return float(s.mean() * sign)


# ─────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────

CSS = """
body{background:#0b0b0f;color:#d1d5db;font-family:'Segoe UI',system-ui,sans-serif;font-size:13px;padding:28px;max-width:1100px;margin:0 auto}
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
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:700px){.grid2{grid-template-columns:1fr}}
.highlight{background:#0a1a0f;border-left:3px solid #166534}
.highlight td:first-child{color:#4ade80}
"""


def _pct_cell(v):
    """Color-coded win-rate table cell."""
    if pd.isna(v) or v is None:
        return '<td class="n">—</td>'
    p = v * 100
    cls = "g" if p >= 60 else ("y" if p >= 55 else ("n" if p >= 48 else "r"))
    return f'<td class="{cls}">{p:.1f}%</td>'


def _ret_cell(v):
    """Color a raw return (already sign-adjusted for direction)."""
    if pd.isna(v) or v is None:
        return '<td class="n">—</td>'
    cls = "g" if v > 0.05 else ("y" if v > 0 else "r")
    return f'<td class="{cls}">{v:+.2f}%</td>'


def summary_table(sigs, title, sub_label=""):
    """Overall long vs short accuracy table."""
    rows = ""
    for direction, side, opt in [("long", "Long (Calls)", "↑"), ("short", "Short (Puts)", "↓")]:
        wrs  = [winrate(sigs, direction, w) for w in WINDOWS]
        rets = [avg_ret(sigs, direction, w)  for w in WINDOWS]
        n    = wrs[0][1]
        wr_cells  = "".join(_pct_cell(wr) for wr, _ in wrs)
        ret_cells = "".join(_ret_cell(r)  for r in rets)
        rows += f'<tr><td class="sym">{opt} {side}</td><td>{n}</td>{wr_cells}{ret_cells}</tr>\n'
    sub = f"<p class='sub'>{sub_label}</p>" if sub_label else ""
    return f"""<h3>{title}</h3>{sub}<div class="card"><table>
<tr>
  <th>Side</th><th>Signals</th>
  <th>Win% +15m</th><th>Win% +30m</th><th>Win% +60m</th>
  <th>Avg +15m</th><th>Avg +30m</th><th>Avg +60m</th>
</tr>
{rows}</table></div>"""


def time_of_day_table(sigs, direction, title):
    """Win rate by hour of day for one direction."""
    sub  = sigs[sigs["direction"] == direction]
    rows = ""
    for slot, grp in sorted(sub.groupby("hour_slot")):
        wr_cells  = ""
        ret_cells = ""
        for w in WINDOWS:
            s = grp[f"up_{w}"].dropna()
            n = len(s)
            if n < MIN_SAMPLES:
                wr_cells  += '<td class="n">—</td>'
                ret_cells += '<td class="n">—</td>'
            else:
                acc = s.mean() if direction == "long" else (1 - s.mean())
                wr_cells += _pct_cell(acc)
                sign = 1 if direction == "long" else -1
                ret_cells += _ret_cell(float(grp[f"ret_{w}"].mean() * sign))
        n_tot = len(grp)
        rows  += f'<tr><td class="sym">{slot}</td><td>{n_tot}</td>{wr_cells}{ret_cells}</tr>\n'
    if not rows:
        return f"<h3>{title}</h3><p style='color:#4b5563;font-size:12px'>No signals with enough data.</p>"
    return f"""<h3>{title}</h3><div class="card"><table>
<tr>
  <th>Hour (ET)</th><th>Signals</th>
  <th>Win% +15m</th><th>Win% +30m</th><th>Win% +60m</th>
  <th>Avg +15m</th><th>Avg +30m</th><th>Avg +60m</th>
</tr>
{rows}</table></div>"""


def recent_signals_table(sigs, n=30):
    """Show the most recent N signals as a trade log."""
    sub  = sigs.tail(n).copy()
    rows = ""
    for ts, row in sub.iterrows():
        d    = row["direction"]
        side = "↑ Long" if d == "long" else "↓ Short"
        cls  = "g" if d == "long" else "r"
        ret_cells = ""
        for w in WINDOWS:
            ret_cells += _ret_cell(
                float(row[f"ret_{w}"]) * (1 if d == "long" else -1)
                if not pd.isna(row[f"ret_{w}"]) else float("nan")
            )
        rows += (
            f'<tr>'
            f'<td class="sym">{ts.strftime("%Y-%m-%d")}</td>'
            f'<td>{ts.strftime("%H:%M")}</td>'
            f'<td class="{cls}">{side}</td>'
            f'<td>${row["price"]:.2f}</td>'
            f'<td>${row["ema200"]:.2f}</td>'
            f'<td>{row["ema_dist_pct"]:+.2f}%</td>'
            f'<td>{row["rel_vol"]:.1f}×</td>'
            f'{ret_cells}'
            f'</tr>\n'
        )
    return f"""<h3>Most Recent {n} Signals</h3><div class="card"><table>
<tr>
  <th>Date</th><th>Time</th><th>Side</th><th>Price</th><th>EMA 200</th>
  <th>Dist</th><th>Vol</th>
  <th>Ret +15m</th><th>Ret +30m</th><th>Ret +60m</th>
</tr>
{rows}</table></div>"""


def build_report(sigs):
    now_str   = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    n_long    = int((sigs["direction"] == "long").sum())
    n_short   = int((sigs["direction"] == "short").sum())
    n_total   = len(sigs)
    vol_sigs  = sigs[sigs["rel_vol"] >= 1.0]

    body  = ""

    # ── Overall accuracy ──
    body += "<h2>Overall Accuracy</h2>"
    body += "<p class='sub'>All EMA-200 pullback bounce signals, direction accuracy at each hold window.</p>"
    body += summary_table(sigs, "All signals")
    body += summary_table(
        vol_sigs, "With volume ≥ day average",
        sub_label="Higher-conviction signals where the bounce bar volume exceeded the session's average 5-min bar volume."
    )

    # ── Time of day breakdown ──
    body += "<h2>Time-of-Day Breakdown</h2>"
    body += "<p class='sub'>Some hours have more reliable bounces than others.</p>"
    body += "<div class='grid2'>"
    body += time_of_day_table(sigs, "long",  "Long (Calls) by Hour")
    body += time_of_day_table(sigs, "short", "Short (Puts) by Hour")
    body += "</div>"

    # ── Recent signal log ──
    body += "<h2>Recent Signal Log</h2>"
    body += recent_signals_table(sigs, n=40)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>EMA 200 Pullback Backtest — {SYMBOL}</title>
<style>{CSS}</style></head><body>
<h1>EMA 200 Pullback Bounce — {SYMBOL} (5-min bars)</h1>
<div class="meta">
  Generated {now_str} &nbsp;·&nbsp; {LOOKBACK}-day lookback &nbsp;·&nbsp;
  {n_total} total signals ({n_long} long / {n_short} short) &nbsp;·&nbsp;
  EMA period: {EMA_PERIOD} &nbsp;·&nbsp; Slope window: {SLOPE_BARS} bars &nbsp;·&nbsp;
  Touch proximity: ±{PROXIMITY*100:.1f}%
</div>

<p class='sub'>
  <strong style="color:#9ca3af">Signal rules:</strong>
  <strong style="color:#4ade80">Long</strong> — EMA 200 rising, prior bar above EMA, current bar's low
  tags the EMA, closes back above it on a green candle.<br>
  <strong style="color:#f87171">Short</strong> — EMA 200 falling, prior bar below EMA, current bar's high
  tags the EMA, closes back below it on a red candle.<br>
  Win = SPY closed in the trade's direction at the measured window.
</p>

{body}

<div class="disc">
  <strong>How to read:</strong> Win% is the fraction of signals where price moved in the
  trade's direction at each hold window (50% = coin flip).
  <span style="color:#4ade80">≥60% = meaningful edge</span>,
  <span style="color:#facc15">55–59% = slight edge</span>,
  gray ≈ noise, <span style="color:#f87171">&lt;48% = inverse edge</span>.
  "Avg" is the mean SPY % move in the trade's direction (sign-adjusted).
  Min {MIN_SAMPLES} signals required per row. &nbsp;·&nbsp;
  <strong>This does NOT simulate stops or profit targets.</strong>
  Direction accuracy ≠ profitability. Past results don't predict the future.
</div>
</body></html>"""


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    if not API_KEY or not API_SECRET:
        sys.exit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")

    print(f"EMA-200 Pullback Bounce Backtest — {SYMBOL} — {LOOKBACK} days — 5-min bars")
    print("=" * 65)

    df = fetch_bars(SYMBOL, TimeFrame(5, TimeFrameUnit.Minute), LOOKBACK, "5m")

    # Market-hours only
    df = df[
        ((df.index.hour == 9)  & (df.index.minute >= 30)) |
        ((df.index.hour >= 10) & (df.index.hour < 16))
    ].copy()
    print(f"  {len(df):,} market-hours bars")

    print("  computing EMA 200 and indicators…")
    df = add_indicators(df)
    df = add_outcomes(df)

    print("  detecting EMA-200 bounce signals…")
    sigs = detect_signals(df)
    if sigs.empty:
        print("No signals found — check your data range or config.")
        return

    n_long  = int((sigs["direction"] == "long").sum())
    n_short = int((sigs["direction"] == "short").sum())
    print(f"  {len(sigs)} signals  ({n_long} long / {n_short} short)")

    # Quick console summary
    print()
    for direction in ("long", "short"):
        print(f"  {direction.upper()}")
        for w in WINDOWS:
            wr, n = winrate(sigs, direction, w)
            if not pd.isna(wr):
                ar = avg_ret(sigs, direction, w)
                print(f"    +{w:2d}m  win={wr*100:.1f}%  avg={ar:+.2f}%  n={n}")
    print()

    out = RESULTS_DIR / "ema200_report.html"
    out.write_text(build_report(sigs), encoding="utf-8")
    print(f"Report written → {out.resolve()}")


if __name__ == "__main__":
    main()
