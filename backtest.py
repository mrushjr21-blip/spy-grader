#!/usr/bin/env python3
"""
backtest.py — Engulfing + SMA10 + MACD + Double Bottom/Top Setup

Win conditions:
  Bullish: price HIGH exceeds signal-bar close at any point from signal time to 12:00pm ET
  Bearish: price LOW drops below signal-bar close at any point from 3:00pm to 4:00pm ET
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

ET          = pytz.timezone("US/Eastern")
API_KEY     = os.environ.get("ALPACA_API_KEY", "")
API_SECRET  = os.environ.get("ALPACA_SECRET_KEY", "")

SYMBOLS          = ["SPY", "MU", "NVDA", "AMD"]
LOOKBACK         = 365
DB_BARS          = 20
DB_TOL           = 0.0025
MIN_SAMPLES      = 10
MIN_SAMPLES_SYM  = 8

CACHE_DIR   = Path("backtest_cache")
RESULTS_DIR = Path("backtest_results")
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Data fetching
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
# Indicators
# ─────────────────────────────────────────────

def add_indicators(df):
    df = df.copy()
    df["sma10"] = df["close"].rolling(10, min_periods=1).mean()
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]      = ema12 - ema26
    df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]
    df["date"] = df.index.date
    daily_open = df.groupby("date")["open"].first()
    daily_high = df.groupby("date")["high"].max()
    daily_low  = df.groupby("date")["low"].min()
    dates = sorted(daily_open.index)
    gap_map = {}
    for i in range(1, len(dates)):
        td, yd = dates[i], dates[i-1]
        o = daily_open[td]
        gap_map[td] = "up" if o > daily_high[yd] else ("down" if o < daily_low[yd] else "none")
    df["gap_type"] = df["date"].map(gap_map).fillna("none")
    df["rel_vol"]  = df["volume"] / df.groupby("date")["volume"].transform("mean")
    df.drop(columns=["date"], inplace=True)
    return df


def add_outcomes(df):
    """
    Per-bar forward-looking outcomes:
      max_high_to_noon     — max high from this bar forward to 12pm same day (bullish win check)
      min_low_3pm_to_close — min low from this bar forward to day end, 3pm+ only (bearish win check)
    """
    df = df.copy()
    df["max_high_to_noon"]     = np.nan
    df["min_low_3pm_to_close"] = np.nan

    for date in pd.unique(df.index.date):
        day = df[df.index.date == date]

        # Bullish: bars before noon — rolling max high going forward to noon
        pre_noon = day[day.index.hour < 12]
        if len(pre_noon) > 0:
            rolling_max = pre_noon["high"].iloc[::-1].cummax().iloc[::-1]
            df.loc[rolling_max.index, "max_high_to_noon"] = rolling_max.values

        # Bearish: bars at 3pm+ — rolling min low going forward to close
        pm = day[day.index.hour >= 15]
        if len(pm) > 0:
            rolling_min = pm["low"].iloc[::-1].cummin().iloc[::-1]
            df.loc[rolling_min.index, "min_low_3pm_to_close"] = rolling_min.values

    return df


# ─────────────────────────────────────────────
# Double bottom / top detection
# ─────────────────────────────────────────────

def _local_lows(lows, swing=2):
    idx = []
    for i in range(swing, len(lows) - swing):
        if all(lows[i] <= lows[i-j] for j in range(1, swing+1)) and \
           all(lows[i] <= lows[i+j] for j in range(1, swing+1)):
            idx.append(i)
    return idx


def _local_highs(highs, swing=2):
    idx = []
    for i in range(swing, len(highs) - swing):
        if all(highs[i] >= highs[i-j] for j in range(1, swing+1)) and \
           all(highs[i] >= highs[i+j] for j in range(1, swing+1)):
            idx.append(i)
    return idx


def check_double_bottom(window_lows, tol=DB_TOL):
    arr  = list(window_lows)
    idxs = _local_lows(arr, swing=2)
    if len(idxs) < 2:
        return False, np.nan
    vals = [arr[i] for i in idxs]
    for a in range(len(vals)):
        for b in range(a+1, len(vals)):
            if abs(vals[a] - vals[b]) / ((vals[a] + vals[b]) / 2) <= tol:
                return True, min(vals[a], vals[b])
    return False, np.nan


def check_double_top(window_highs, tol=DB_TOL):
    arr  = list(window_highs)
    idxs = _local_highs(arr, swing=2)
    if len(idxs) < 2:
        return False, np.nan
    vals = [arr[i] for i in idxs]
    for a in range(len(vals)):
        for b in range(a+1, len(vals)):
            if abs(vals[a] - vals[b]) / ((vals[a] + vals[b]) / 2) <= tol:
                return True, max(vals[a], vals[b])
    return False, np.nan


# ─────────────────────────────────────────────
# Signal detection
# ─────────────────────────────────────────────

def detect_signals(df):
    rows = []

    close      = df["close"].values
    open_      = df["open"].values
    high       = df["high"].values
    low        = df["low"].values
    sma10      = df["sma10"].values
    mhist      = df["macd_hist"].values
    gap        = df["gap_type"].values
    rvol       = df["rel_vol"].values
    max_h_noon = df["max_high_to_noon"].values
    min_l_pm   = df["min_low_3pm_to_close"].values

    for i in range(DB_BARS + 2, len(df) - 1):
        c, o   = close[i],  open_[i]
        pc, po = close[i-1], open_[i-1]
        s, ps  = sma10[i],  sma10[i-1]
        h, ph  = mhist[i],  mhist[i-1]

        body_lo  = min(o, c);   body_hi  = max(o, c)
        pbody_lo = min(po, pc); pbody_hi = max(po, pc)
        bull_engulf = (c > o) and (body_lo <= pbody_lo) and (body_hi >= pbody_hi)
        bear_engulf = (c < o) and (body_lo <= pbody_lo) and (body_hi >= pbody_hi)

        sma_cross_bull = (pc < ps) and (c > s)
        sma_cross_bear = (pc > ps) and (c < s)

        macd_curl_bull = (h > ph) or (h > 0)
        macd_curl_bear = (h < ph) or (h < 0)

        if not (sma_cross_bull or sma_cross_bear):
            continue

        win_low  = low[i - DB_BARS : i]
        win_high = high[i - DB_BARS : i]
        db_found, support    = check_double_bottom(win_low)
        dt_found, resistance = check_double_top(win_high)

        ts        = df.index[i]
        time_slot = f"{ts.hour:02d}:00"
        gap_t     = gap[i]
        rv        = rvol[i]

        if sma_cross_bull:
            mh  = max_h_noon[i]
            won = bool(mh > c) if not np.isnan(mh) else np.nan
            rows.append({
                "timestamp": ts,
                "direction": "bullish",
                "price":     c,
                "engulf":    bull_engulf,
                "macd_ok":   macd_curl_bull,
                "db_found":  db_found,
                "support":   support,
                "time_slot": time_slot,
                "gap_type":  gap_t,
                "rel_vol":   rv,
                "won":       won,
            })

        if sma_cross_bear:
            ml  = min_l_pm[i]
            won = bool(ml < c) if not np.isnan(ml) else np.nan
            rows.append({
                "timestamp": ts,
                "direction": "bearish",
                "price":     c,
                "engulf":    bear_engulf,
                "macd_ok":   macd_curl_bear,
                "db_found":  dt_found,
                "support":   resistance,
                "time_slot": time_slot,
                "gap_type":  gap_t,
                "rel_vol":   rv,
                "won":       won,
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("timestamp")


# ─────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────

def win_rate(sigs, direction):
    sub = sigs[sigs["direction"] == direction]["won"].dropna()
    if len(sub) < MIN_SAMPLES:
        return np.nan, 0
    return sub.mean(), len(sub)


def variant_table(sigs):
    full = sigs["engulf"] & sigs["macd_ok"] & sigs["db_found"]
    variants = {
        "SMA10 cross only":        sigs,
        "SMA + Engulfing":         sigs[sigs["engulf"]],
        "SMA + MACD curl":         sigs[sigs["macd_ok"]],
        "SMA + Double Bottom/Top": sigs[sigs["db_found"]],
        "Full setup (all 4)":      sigs[full],
        "Full setup + Vol ≥0.8×":  sigs[full & (sigs["rel_vol"] >= 0.8)],
    }
    rows = []
    for label, sub in variants.items():
        wr_bull, n_bull = win_rate(sub, "bullish")
        wr_bear, n_bear = win_rate(sub, "bearish")
        rows.append({"variant": label, "n_bull": n_bull, "n_bear": n_bear,
                     "wr_bull": wr_bull, "wr_bear": wr_bear})
    return pd.DataFrame(rows)


def condition_win_rate(full_sigs, col, direction, min_vol=None):
    mask = (
        full_sigs["engulf"] & full_sigs["macd_ok"] & full_sigs["db_found"] &
        (full_sigs["direction"] == direction)
    )
    if min_vol is not None:
        mask = mask & (full_sigs["rel_vol"] >= min_vol)
    sub = full_sigs[mask][[col, "won"]].dropna()
    if len(sub) == 0:
        return pd.DataFrame()
    grp = sub.groupby(col)["won"].agg(["mean", "count"])
    grp.columns = ["win_rate", "n"]
    return grp[grp["n"] >= MIN_SAMPLES].sort_values("win_rate", ascending=False)[["n", "win_rate"]]


def symbol_ranking(all_sigs, direction, time_slots):
    full = all_sigs[all_sigs["engulf"] & all_sigs["macd_ok"] & all_sigs["db_found"]]
    sub  = full[(full["direction"] == direction) & (full["time_slot"].isin(time_slots))]
    rows = []
    for sym, grp in sub.groupby("symbol"):
        valid = grp["won"].dropna()
        if len(valid) < MIN_SAMPLES_SYM:
            continue
        rows.append({"symbol": sym, "n": len(valid), "win_rate": valid.mean()})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("win_rate", ascending=False)


# ─────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────

CSS = """
body{background:#0b0b0f;color:#d1d5db;font-family:'Segoe UI',system-ui,sans-serif;font-size:13px;padding:28px;max-width:1100px;margin:0 auto}
h1{font-size:22px;font-weight:700;color:#f9fafb;margin:0 0 4px}
h2{font-size:12px;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:.1em;margin:32px 0 12px;border-bottom:1px solid #1f2028;padding-bottom:8px}
h3{font-size:13px;font-weight:600;color:#9ca3af;margin:20px 0 6px}
p.sub{color:#6b7280;font-size:12px;margin:0 0 12px}
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
"""

def pct(v):
    if pd.isna(v):
        return '<td class="n">—</td>'
    p = v * 100
    c = "g" if p >= 60 else ("y" if p >= 55 else ("n" if p >= 48 else "r"))
    return f'<td class="{c}">{p:.1f}%</td>'


def variant_html(vt):
    rows = ""
    for _, r in vt.iterrows():
        rows += f'<tr><td class="sym">{r["variant"]}</td>'
        rows += f'<td>{int(r["n_bull"])} bull / {int(r["n_bear"])} bear</td>'
        rows += pct(r["wr_bull"]) + pct(r["wr_bear"])
        rows += "</tr>"
    return f"""<div class="card"><table>
<tr><th>Variant</th><th>Signal Count</th><th>Bull Win %</th><th>Bear Win %</th></tr>
{rows}</table></div>"""


def cond_html(df_cond, title, direction):
    if df_cond is None or len(df_cond) == 0:
        return f"<h3>{title}</h3><p style='color:#4b5563;font-size:12px'>Not enough data (min {MIN_SAMPLES} signals).</p>"
    rows = ""
    for idx, row in df_cond.iterrows():
        rows += f'<tr><td class="sym">{idx}</td><td>{int(row["n"])}</td>{pct(row["win_rate"])}</tr>'
    return f"""<h3>{title}</h3><div class="card"><table>
<tr><th>Condition</th><th>Signals</th><th>Win % ({direction})</th></tr>
{rows}</table></div>"""


def symbol_ranking_html(df_rank, title):
    if df_rank is None or len(df_rank) == 0:
        return f"<h3>{title}</h3><p style='color:#4b5563;font-size:12px'>Not enough data.</p>"
    rows = ""
    for i, (_, r) in enumerate(df_rank.iterrows()):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
        rows += f'<tr><td class="sym">{medal} {r["symbol"]}</td><td>{int(r["n"])}</td>{pct(r["win_rate"])}</tr>'
    return f"""<h3>{title}</h3><div class="card"><table>
<tr><th>Symbol</th><th>Signals</th><th>Win %</th></tr>
{rows}</table></div>"""


def build_report(all_sigs, symbol_counts):
    now_str = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")

    full      = all_sigs[all_sigs["engulf"] & all_sigs["macd_ok"] & all_sigs["db_found"]]
    full_bull = full[full["direction"] == "bullish"]
    full_bear = full[full["direction"] == "bearish"]

    vt = variant_table(all_sigs)

    body  = "<h2>Symbol Rankings — Bullish (10–11am)</h2>"
    body += "<p class='sub'>Full setup fired 10–11am. Win = price high exceeded signal close before 12pm. Min 8 signals.</p>"
    body += symbol_ranking_html(symbol_ranking(all_sigs, "bullish", ["10:00", "11:00"]), "Bullish Win Rate")

    body += "<h2>Symbol Rankings — Bearish (3pm)</h2>"
    body += "<p class='sub'>Full setup fired at 3pm. Win = price low dropped below signal close before 4pm. Min 8 signals.</p>"
    body += symbol_ranking_html(symbol_ranking(all_sigs, "bearish", ["15:00"]), "Bearish Win Rate")

    body += "<h2>Filter Variants — What Each Condition Adds</h2>"
    body += "<p class='sub'>Bull win: high > signal close before noon. Bear win: low < signal close before 4pm.</p>"
    body += variant_html(vt)

    body += "<h2>Full Setup — Win Rate by Time of Day</h2>"
    body += f"<p class='sub'>{len(full_bull)} bullish · {len(full_bear)} bearish full-setup signals total.</p>"
    body += "<div class='grid2'>"
    body += cond_html(condition_win_rate(all_sigs, "time_slot", "bullish"), "Bullish Win % by Hour", "bullish")
    body += cond_html(condition_win_rate(all_sigs, "time_slot", "bearish"), "Bearish Win % by Hour", "bearish")
    body += "</div>"

    body += "<h2>Full Setup — Win Rate by Gap Type</h2>"
    body += "<div class='grid2'>"
    body += cond_html(condition_win_rate(all_sigs, "gap_type", "bullish"), "Bullish Win % by Gap", "bullish")
    body += cond_html(condition_win_rate(all_sigs, "gap_type", "bearish"), "Bearish Win % by Gap", "bearish")
    body += "</div>"

    body += "<h2>Full Setup — Win Rate by Relative Volume</h2>"
    all_sigs_rv = all_sigs.copy()
    all_sigs_rv["vol_bucket"] = pd.cut(
        all_sigs_rv["rel_vol"],
        bins=[0, 0.8, 1.2, 1.5, 2.0, 99],
        labels=["<0.8×", "0.8–1.2×", "1.2–1.5×", "1.5–2.0×", ">2.0×"]
    )
    body += "<div class='grid2'>"
    body += cond_html(condition_win_rate(all_sigs_rv, "vol_bucket", "bullish"), "Bullish Win % by Volume", "bullish")
    body += cond_html(condition_win_rate(all_sigs_rv, "vol_bucket", "bearish"), "Bearish Win % by Volume", "bearish")
    body += "</div>"

    counts_str = " &nbsp;·&nbsp; ".join(f"{s}: {n}" for s, n in symbol_counts.items())
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Setup Backtest</title>
<style>{CSS}</style></head><body>
<h1>Engulfing + SMA10 + MACD + Double Bottom/Top — Backtest</h1>
<div class="meta">
  {now_str} &nbsp;·&nbsp; {LOOKBACK} days &nbsp;·&nbsp; 5-minute bars &nbsp;·&nbsp; {counts_str}
</div>
{body}
<div class="disc">
<strong>Win conditions:</strong>
Bullish: any bar high exceeds signal-bar close between signal time and 12:00pm ET on the same day.
Bearish: any bar low drops below signal-bar close between 3:00pm and 4:00pm ET on the same day.
<br><br>
<strong>How to read win %:</strong>
% of full-setup signals where price reached the target direction within the window.
50% = coin flip. <span style="color:#4ade80">≥60% = meaningful edge</span>,
<span style="color:#facc15">55–60% = slight edge</span>, gray = noise,
<span style="color:#f87171">&lt;48% = works against you</span>.
Minimum {MIN_SAMPLES} signals per row.
<br><br>
<strong>Limitation:</strong> {LOOKBACK} days of data. Past win rate does not guarantee future results.
</div>
</body></html>"""


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    if not API_KEY or not API_SECRET:
        sys.exit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file.")

    print(f"Setup Backtest — {LOOKBACK} days — 5-minute bars — {len(SYMBOLS)} symbols")
    print("=" * 60)

    all_sigs_list = []
    symbol_counts = {}

    for symbol in SYMBOLS:
        print(f"\n{symbol}")
        df = fetch_bars(symbol, TimeFrame(5, TimeFrameUnit.Minute), LOOKBACK, "5m")

        df = df[
            ((df.index.hour == 9) & (df.index.minute >= 30)) |
            ((df.index.hour >= 10) & (df.index.hour < 16))
        ].copy()

        print(f"  {len(df):,} bars · computing indicators…")
        df = add_indicators(df)
        df = add_outcomes(df)

        print("  detecting signals…")
        sigs = detect_signals(df)
        if len(sigs):
            sigs["symbol"] = symbol
            full_n = int((sigs["engulf"] & sigs["macd_ok"] & sigs["db_found"]).sum())
            symbol_counts[symbol] = full_n
            all_sigs_list.append(sigs)
            print(f"  SMA crosses: {len(sigs)} · full setup signals: {full_n}")
        else:
            symbol_counts[symbol] = 0
            print("  no signals found")

    if not all_sigs_list:
        print("No signals found across any symbol.")
        return

    print("\nBuilding report…")
    all_sigs = pd.concat(all_sigs_list)

    report = build_report(all_sigs, symbol_counts)
    out = RESULTS_DIR / "report.html"
    out.write_text(report, encoding="utf-8")
    print(f"\nDone. Open: {out.resolve()}")


if __name__ == "__main__":
    main()
