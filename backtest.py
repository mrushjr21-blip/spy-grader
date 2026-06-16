#!/usr/bin/env python3
"""
backtest.py — Opportunity Win Backtest

Win condition (matches active day-trading style):
  Bullish: any bar HIGH exceeds signal-bar close × (1 + LEEWAY) before 12:00pm ET
  Bearish: any bar LOW drops below signal-bar close × (1 − LEEWAY) before 4:00pm ET

If price touched your target at any point in the window, you had a chance to take profit.
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

SYMBOLS         = ["SPY", "MU", "NVDA"]
LOOKBACK        = 365
DB_BARS         = 20
DB_TOL          = 0.0025
MIN_SAMPLES     = 10
MIN_SAMPLES_SYM = 8
BULL_HOURS      = {"10:00", "11:00"}
BEAR_HOURS      = {"15:00"}
VOL_FILTER      = 1.0
LEEWAY          = 0.001   # 0.1% — price must exceed signal close by this margin to count as a win

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

    close = df["close"].values
    open_ = df["open"].values
    high  = df["high"].values
    low   = df["low"].values
    sma10 = df["sma10"].values
    mhist = df["macd_hist"].values
    gap   = df["gap_type"].values
    rvol  = df["rel_vol"].values

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
            noon = ts.normalize() + pd.Timedelta(hours=12)
            fwd  = df[(df.index > ts) & (df.index <= noon)]
            won  = bool((fwd["high"] > c * (1 + LEEWAY)).any()) if len(fwd) > 0 else np.nan
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
            close_time = ts.normalize() + pd.Timedelta(hours=16)
            fwd = df[(df.index > ts) & (df.index <= close_time)]
            won = bool((fwd["low"] < c * (1 - LEEWAY)).any()) if len(fwd) > 0 else np.nan
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

def wr(series, min_n=MIN_SAMPLES):
    valid = series.dropna()
    if len(valid) < min_n:
        return np.nan, len(valid)
    return valid.mean(), len(valid)


def full_setup_mask(sigs):
    return sigs["engulf"] & sigs["macd_ok"] & sigs["db_found"]


def variant_table(sigs):
    bull_prime = sigs[(sigs["direction"] == "bullish") & sigs["time_slot"].isin(BULL_HOURS)]
    bear_prime = sigs[(sigs["direction"] == "bearish") & sigs["time_slot"].isin(BEAR_HOURS)]

    def sub_bull(mask): return bull_prime[mask[bull_prime.index]] if len(bull_prime) else bull_prime
    def sub_bear(mask): return bear_prime[mask[bear_prime.index]] if len(bear_prime) else bear_prime

    variants = [
        ("SMA cross only",          sigs,                                              bull_prime,                                              bear_prime),
        ("SMA + Engulfing",         sigs[sigs["engulf"]],                              bull_prime[bull_prime["engulf"]],                         bear_prime[bear_prime["engulf"]]),
        ("SMA + MACD curl",         sigs[sigs["macd_ok"]],                             bull_prime[bull_prime["macd_ok"]],                        bear_prime[bear_prime["macd_ok"]]),
        ("SMA + Double Bottom/Top", sigs[sigs["db_found"]],                            bull_prime[bull_prime["db_found"]],                       bear_prime[bear_prime["db_found"]]),
        ("SMA + Engulf + MACD",     sigs[sigs["engulf"] & sigs["macd_ok"]],            bull_prime[bull_prime["engulf"] & bull_prime["macd_ok"]], bear_prime[bear_prime["engulf"] & bear_prime["macd_ok"]]),
        ("SMA + Engulf + DB/DT",    sigs[sigs["engulf"] & sigs["db_found"]],           bull_prime[bull_prime["engulf"] & bull_prime["db_found"]],bear_prime[bear_prime["engulf"] & bear_prime["db_found"]]),
        ("SMA + MACD + DB/DT",      sigs[sigs["macd_ok"] & sigs["db_found"]],          bull_prime[bull_prime["macd_ok"] & bull_prime["db_found"]],bear_prime[bear_prime["macd_ok"] & bear_prime["db_found"]]),
        ("Full setup (all 4)",      sigs[full_setup_mask(sigs)],                       bull_prime[full_setup_mask(bull_prime)],                  bear_prime[full_setup_mask(bear_prime)]),
    ]
    rows = []
    for label, _all, bp, brp in variants:
        wr_b, n_b = wr(bp["won"])
        wr_r, n_r = wr(brp["won"])
        rows.append({"variant": label, "n_bull": n_b, "n_bear": n_r,
                     "wr_bull": wr_b, "wr_bear": wr_r})
    return pd.DataFrame(rows)


def symbol_table(sigs, direction, time_slots):
    full = sigs[full_setup_mask(sigs)]
    sub  = full[(full["direction"] == direction) & (full["time_slot"].isin(time_slots))]
    rows = []
    for sym, grp in sub.groupby("symbol"):
        w, n = wr(grp["won"], min_n=MIN_SAMPLES_SYM)
        if np.isnan(w):
            continue
        rows.append({"symbol": sym, "n": n, "win_rate": w})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("win_rate", ascending=False)


def time_table(sigs, direction):
    full = sigs[full_setup_mask(sigs) & (sigs["direction"] == direction)]
    rows = []
    for slot, grp in full.groupby("time_slot"):
        w, n = wr(grp["won"])
        if np.isnan(w):
            continue
        rows.append({"time": slot, "n": n, "win_rate": w})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("time")


def overall_table(sigs, direction, time_slots):
    full = sigs[full_setup_mask(sigs)]
    sub  = full[(full["direction"] == direction) & (full["time_slot"].isin(time_slots))]
    rows = []
    for label, mask in [
        (f"{'Bullish' if direction=='bullish' else 'Bearish'} — all volume", sub),
        (f"{'Bullish' if direction=='bullish' else 'Bearish'} — vol ≥{VOL_FILTER}×", sub[sub["rel_vol"] >= VOL_FILTER]),
    ]:
        w, n = wr(mask["won"])
        if not np.isnan(w):
            rows.append({"label": label, "n": n, "win_rate": w})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# HTML helpers
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
th.r{text-align:right}
td{padding:7px 12px;border-bottom:1px solid #141420;font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:none}
tr:hover td{background:#13131e}
.g{color:#4ade80;font-weight:600}.y{color:#facc15}.gr{color:#6b7280}.r{color:#f87171;font-weight:600}
.sym{font-weight:700;color:#f9fafb}
.disc{background:#0e0e18;border:1px solid #1f2028;border-radius:8px;padding:14px 16px;color:#4b5563;font-size:12px;line-height:1.7;margin-top:36px}
.disc strong{color:#6b7280}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:700px){.grid2{grid-template-columns:1fr}}
"""

def pct_td(v):
    if pd.isna(v):
        return '<td class="gr" style="text-align:right">—</td>'
    p = v * 100
    c = "g" if p >= 60 else ("y" if p >= 55 else ("gr" if p >= 48 else "r"))
    return f'<td class="{c}" style="text-align:right">{p:.1f}%</td>'


def overall_html(bull_df, bear_df):
    rows = ""
    for df in [bull_df, bear_df]:
        for _, r in df.iterrows():
            rows += f'<tr><td class="sym">{r["label"]}</td>'
            rows += f'<td style="text-align:right">{int(r["n"])}</td>'
            rows += pct_td(r["win_rate"]) + "</tr>"
    return f"""<div class="card"><table>
<tr><th>Filter</th><th class="r">Signals</th><th class="r">Win %</th></tr>
{rows}</table></div>"""


def variant_html(vt):
    rows = ""
    for _, r in vt.iterrows():
        rows += f'<tr><td class="sym">{r["variant"]}</td>'
        rows += f'<td style="text-align:right">{int(r["n_bull"])} bull / {int(r["n_bear"])} bear</td>'
        rows += pct_td(r["wr_bull"]) + pct_td(r["wr_bear"]) + "</tr>"
    return f"""<div class="card"><table>
<tr><th>Variant</th><th class="r">Signals</th><th class="r">Bull Win %</th><th class="r">Bear Win %</th></tr>
{rows}</table></div>"""


def symbol_html(df, title):
    if df is None or len(df) == 0:
        return f"<h3>{title}</h3><p style='color:#4b5563;font-size:12px'>Not enough data (min {MIN_SAMPLES_SYM} signals).</p>"
    rows = ""
    for i, (_, r) in enumerate(df.iterrows()):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
        rows += f'<tr><td class="sym">{medal} {r["symbol"]}</td>'
        rows += f'<td style="text-align:right">{int(r["n"])}</td>'
        rows += pct_td(r["win_rate"]) + "</tr>"
    return f"""<h3>{title}</h3><div class="card"><table>
<tr><th>Symbol</th><th class="r">Signals</th><th class="r">Win %</th></tr>
{rows}</table></div>"""


def time_html(df, title):
    if df is None or len(df) == 0:
        return f"<h3>{title}</h3><p style='color:#4b5563;font-size:12px'>Not enough data.</p>"
    rows = ""
    for _, r in df.iterrows():
        rows += f'<tr><td class="sym">{r["time"]}</td>'
        rows += f'<td style="text-align:right">{int(r["n"])}</td>'
        rows += pct_td(r["win_rate"]) + "</tr>"
    return f"""<h3>{title}</h3><div class="card"><table>
<tr><th>Time</th><th class="r">Signals</th><th class="r">Win %</th></tr>
{rows}</table></div>"""


# ─────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────

def build_report(all_sigs, symbol_counts):
    now_str   = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    counts_str = " · ".join(f"{s}: {n}" for s, n in symbol_counts.items())

    bull_overall = overall_table(all_sigs, "bullish", BULL_HOURS)
    bear_overall = overall_table(all_sigs, "bearish", BEAR_HOURS)

    vt       = variant_table(all_sigs)
    bull_sym = symbol_table(all_sigs, "bullish", BULL_HOURS)
    bear_sym = symbol_table(all_sigs, "bearish", BEAR_HOURS)
    bull_tod = time_table(all_sigs, "bullish")
    bear_tod = time_table(all_sigs, "bearish")

    body  = "<h2>Overall — Full Setup (all 4 conditions)</h2>"
    body += "<p class='sub'>Prime window only: bullish 10–11am, bearish 3pm. Full setup = SMA cross + engulf + MACD curl + double bottom/top.</p>"
    body += overall_html(bull_overall, bear_overall)

    body += "<h2>What Each Condition Adds (start from SMA only, read down)</h2>"
    body += "<p class='sub'>All times included. Win = close holds on winning side of signal close for >X% of bars to noon (bull) / 4pm (bear).</p>"
    body += variant_html(vt)

    body += "<h2>Per-Symbol Summary — Bullish (sustained to 12pm)</h2>"
    body += "<p class='sub'>Full setup, 10–11am signals only. Min 8 signals per symbol.</p>"
    body += symbol_html(bull_sym, "Bullish Win Rate by Symbol")

    body += "<h2>Per-Symbol Summary — Bearish (sustained to 4pm)</h2>"
    body += "<p class='sub'>Full setup, 3pm signals only. Min 8 signals per symbol.</p>"
    body += symbol_html(bear_sym, "Bearish Win Rate by Symbol")

    body += "<h2>By Time of Day — Bullish (all volume)</h2>"
    body += time_html(bull_tod, "Bullish Win % by Hour")

    body += "<h2>By Time of Day — Bearish (all volume)</h2>"
    body += time_html(bear_tod, "Bearish Win % by Hour")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Sustained Price Hold Backtest</title>
<style>{CSS}</style></head><body>
<h1>Opportunity Win — Full Setup Backtest</h1>
<div class="meta">{now_str} &nbsp;·&nbsp; {LOOKBACK} days &nbsp;·&nbsp; 5-minute bars &nbsp;·&nbsp; {counts_str}</div>
{body}
<div class="disc">
<strong>Win condition:</strong>
Bullish: any bar HIGH exceeds signal-bar close + 0.1% at any point from signal time to 12:00pm ET.
Bearish: any bar LOW drops below signal-bar close − 0.1% at any point from signal time to 4:00pm ET.
Matches active day-trading: if you were watching and price touched profit, you had the opportunity to take it.
<br><br>
<strong>Color:</strong> <span style="color:#4ade80">≥60% green</span> · <span style="color:#facc15">55–60% yellow</span> · gray = noise · <span style="color:#f87171">&lt;48% red</span>.
Minimum {MIN_SAMPLES} signals per row.
<br><br>
<strong>Limitation:</strong> {LOOKBACK} days of IEX data. Past results do not guarantee future performance.
</div>
</body></html>"""


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    if not API_KEY or not API_SECRET:
        sys.exit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file.")

    print(f"Sustained Price Hold Backtest — {LOOKBACK} days — 5-minute bars — {len(SYMBOLS)} symbols")
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

        print("  detecting signals…")
        sigs = detect_signals(df)
        if len(sigs):
            sigs["symbol"] = symbol
            full_n = int(full_setup_mask(sigs).sum())
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

    vt = variant_table(all_sigs)
    print("\n-- Condition Combos (PRIME WINDOW ONLY: 10-11am bull / 3pm bear) --")
    print(f"{'Variant':<30} {'Bull N':>6} {'Bull Win%':>9} {'Bear N':>6} {'Bear Win%':>9}")
    print("-" * 65)
    for _, r in vt.iterrows():
        bw = f"{r['wr_bull']*100:.1f}%" if not pd.isna(r['wr_bull']) else "  —"
        rw = f"{r['wr_bear']*100:.1f}%" if not pd.isna(r['wr_bear']) else "  —"
        print(f"{r['variant']:<30} {int(r['n_bull']):>6} {bw:>9} {int(r['n_bear']):>6} {rw:>9}")

    report = build_report(all_sigs, symbol_counts)
    out = RESULTS_DIR / "report.html"
    out.write_text(report, encoding="utf-8")
    print(f"\nDone. Open: {out.resolve()}")


if __name__ == "__main__":
    main()
