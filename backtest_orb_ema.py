#!/usr/bin/env python3
"""
backtest_orb_ema.py — Opening Range Breakout & 20-EMA Pullback (SPY, 5-minute bars)

Two intraday SPY strategies, backtested over ~1 year. For every signal we measure
whether price moved in the trade's direction 15 / 30 / 60 minutes later
(direction-accuracy — the same style as backtest.py). Stop-losses and profit
targets are NOT simulated in this version.

Strategy 1 — Opening Range Breakout (ORB):
  - Opening range = high/low of the first 15 minutes (09:30–09:45, the first
    three 5-min bars).
  - Long (calls)  on the first 5-min close ABOVE the OR high after 09:45.
  - Short (puts)  on the first 5-min close BELOW the OR low.
  - One signal per day — the first breakout, whichever side breaks first.
  - "With volume" = breakout bar volume >= the day's average 5-min bar volume.

Strategy 2 — Trend Continuation (Pullback to 20 EMA):
  - Uptrend   = 20-EMA rising and price above it; downtrend = mirror image.
  - Long (calls) when price pulls back to touch the 20 EMA and a green 5-min
    candle closes back above it. Short (puts) = mirror image.

Run:  python backtest_orb_ema.py
Output: backtest_results/orb_ema_report.html
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

ET             = pytz.timezone("US/Eastern")
API_KEY        = os.environ.get("ALPACA_API_KEY", "")
API_SECRET     = os.environ.get("ALPACA_SECRET_KEY", "")

SYMBOL         = "SPY"
LOOKBACK       = 365
WINDOWS        = [15, 30, 60]    # forward outcome windows in minutes
EMA_PERIOD     = 20
EMA_SLOPE_BARS = 3              # bars used to gauge EMA direction
MIN_SAMPLES    = 10            # hide table rows below this many signals

CACHE_DIR   = Path("backtest_cache")
RESULTS_DIR = Path("backtest_results")
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Data fetching  (cache key matches backtest.py → reuses SPY_5m_365d.pkl)
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
# Indicators & forward outcomes
# ─────────────────────────────────────────────

def add_indicators(df):
    df = df.copy()
    df["ema20"]   = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    by_day        = df.groupby(df.index.date)
    df["rel_vol"] = df["volume"] / by_day["volume"].transform("mean")
    return df


def add_outcomes(df):
    df = df.copy()
    bars_per_window = {15: 3, 30: 6, 60: 12}   # 5-min bars per window
    for w, bars in bars_per_window.items():
        fwd = df.groupby(df.index.date)["close"].transform(lambda x: x.shift(-bars))
        df[f"ret_{w}"] = (fwd - df["close"]) / df["close"] * 100
        df[f"up_{w}"]  = df[f"ret_{w}"] > 0
    return df


def _row(ts, bar, direction, strategy):
    r = {
        "timestamp": ts,
        "strategy":  strategy,
        "direction": direction,
        "price":     float(bar["close"]),
        "rel_vol":   float(bar["rel_vol"]),
        "time_slot": f"{ts.hour:02d}:00",
    }
    for w in WINDOWS:
        r[f"ret_{w}"] = bar[f"ret_{w}"]
        r[f"up_{w}"]  = bar[f"up_{w}"]
    return r


# ─────────────────────────────────────────────
# Strategy 1 — Opening Range Breakout
# ─────────────────────────────────────────────

def detect_orb(df):
    """First breakout of the 09:30–09:45 range, one signal per day."""
    rows = []
    for _, day in df.groupby(df.index.date):
        day = day.sort_index()
        or_bars = day[(day.index.hour == 9) & (day.index.minute < 45)]
        if len(or_bars) < 1:
            continue
        or_high = or_bars["high"].max()
        or_low  = or_bars["low"].min()
        rest = day[(day.index.hour > 9) | ((day.index.hour == 9) & (day.index.minute >= 45))]
        for ts, bar in rest.iterrows():
            c = float(bar["close"])
            if c > or_high:
                rows.append(_row(ts, bar, "long", "orb"))
                break
            if c < or_low:
                rows.append(_row(ts, bar, "short", "orb"))
                break
    return pd.DataFrame(rows).set_index("timestamp") if rows else pd.DataFrame()


# ─────────────────────────────────────────────
# Strategy 2 — Trend Continuation (pullback to 20 EMA)
# ─────────────────────────────────────────────

def detect_ema_pullback(df):
    """Bounce off a rising/falling 20 EMA. Skips immediately-consecutive repeats."""
    rows = []
    close = df["close"].values
    open_ = df["open"].values
    low   = df["low"].values
    high  = df["high"].values
    ema   = df["ema20"].values
    dates = np.array(df.index.date)
    idx   = df.index
    last  = None   # (direction, bar_index) of the previous signal

    for i in range(EMA_SLOPE_BARS + 1, len(df)):
        if dates[i] != dates[i - EMA_SLOPE_BARS]:
            continue   # need the slope window within one session

        ema_up   = ema[i] > ema[i - EMA_SLOPE_BARS]
        ema_down = ema[i] < ema[i - EMA_SLOPE_BARS]
        c, o = close[i], open_[i]

        long_sig = (ema_up and close[i - 1] > ema[i - 1] and
                    low[i] <= ema[i] and c > o and c > ema[i])
        short_sig = (ema_down and close[i - 1] < ema[i - 1] and
                     high[i] >= ema[i] and c < o and c < ema[i])

        ts = idx[i]
        if long_sig and last != ("long", i - 1):
            rows.append(_row(ts, df.iloc[i], "long", "ema"))
            last = ("long", i)
        elif short_sig and last != ("short", i - 1):
            rows.append(_row(ts, df.iloc[i], "short", "ema"))
            last = ("short", i)

    return pd.DataFrame(rows).set_index("timestamp") if rows else pd.DataFrame()


# ─────────────────────────────────────────────
# Accuracy helpers
# ─────────────────────────────────────────────

def winrate(sub, direction, window):
    col = f"up_{window}"
    s = sub[sub["direction"] == direction][col].dropna()
    if len(s) < MIN_SAMPLES:
        return np.nan, len(s)
    acc = s.mean() if direction == "long" else (1 - s.mean())
    return acc, len(s)


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


def summary_html(sigs, title, vol_filter=False):
    sub = sigs[sigs["rel_vol"] >= 1.0] if vol_filter else sigs
    rows = ""
    for direction, side, opt in [("long", "Long", "Calls"), ("short", "Short", "Puts")]:
        cells = "".join(pct(winrate(sub, direction, w)[0]) for w in WINDOWS)
        _, n = winrate(sub, direction, 15)
        rows += f'<tr><td class="sym">{side} ({opt})</td><td>{n}</td>{cells}</tr>'
    return f"""<h3>{title}</h3><div class="card"><table>
<tr><th>Side</th><th>Signals</th><th>Win +15m</th><th>Win +30m</th><th>Win +60m</th></tr>
{rows}</table></div>"""


def time_of_day_html(sigs, direction, title):
    sub = sigs[sigs["direction"] == direction]
    rows = ""
    for slot, grp in sub.groupby("time_slot"):
        cells = ""
        for w in WINDOWS:
            s = grp[f"up_{w}"].dropna()
            if len(s) < MIN_SAMPLES:
                cells += '<td class="n">—</td>'
            else:
                acc = s.mean() if direction == "long" else (1 - s.mean())
                cells += pct(acc)
        n = len(grp["up_15"].dropna())
        rows += f'<tr><td class="sym">{slot}</td><td>{n}</td>{cells}</tr>'
    if not rows:
        return f"<h3>{title}</h3><p style='color:#4b5563;font-size:12px'>No signals.</p>"
    return f"""<h3>{title}</h3><div class="card"><table>
<tr><th>Time Slot</th><th>Signals</th><th>Win +15m</th><th>Win +30m</th><th>Win +60m</th></tr>
{rows}</table></div>"""


def build_report(orb, ema):
    now_str = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")

    body  = "<h2>Strategy 1 — Opening Range Breakout (09:30–09:45)</h2>"
    body += ("<p class='sub'>First breakout of the opening-range high (Long/Calls) or low "
             "(Short/Puts), one signal per day. Win = SPY closed in the trade's direction at "
             "each window.</p>")
    body += summary_html(orb, "All first breakouts")
    body += summary_html(orb, "Breakouts with volume ≥ day average", vol_filter=True)

    body += "<h2>Strategy 2 — Trend Continuation (Pullback to 20 EMA)</h2>"
    body += ("<p class='sub'>Bounce off a rising 20 EMA (Long/Calls) or falling 20 EMA "
             "(Short/Puts) on the 5-min chart.</p>")
    body += summary_html(ema, "All pullback bounces")
    body += "<div class='grid2'>"
    body += time_of_day_html(ema, "long",  "Long (Calls) by Time of Day")
    body += time_of_day_html(ema, "short", "Short (Puts) by Time of Day")
    body += "</div>"

    n_orb = len(orb)
    n_ema = len(ema)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>ORB & 20-EMA Backtest — SPY</title>
<style>{CSS}</style></head><body>
<h1>Opening Range Breakout & 20-EMA Pullback — {SYMBOL}</h1>
<div class="meta">
  {now_str} &nbsp;·&nbsp; {LOOKBACK} days &nbsp;·&nbsp; 5-minute bars
  &nbsp;·&nbsp; ORB signals: {n_orb} &nbsp;·&nbsp; EMA signals: {n_ema}
</div>
{body}
<div class="disc">
<strong>How to read accuracy:</strong>
% of signals where price moved in the trade's direction at each window.
50% = coin flip. <span style="color:#4ade80">≥60% = meaningful edge</span>,
<span style="color:#facc15">55–60% = slight edge</span>, gray = noise,
<span style="color:#f87171">&lt;48% = works against you</span>.
Minimum {MIN_SAMPLES} signals required per row.
<br><br>
<strong>Important limitation:</strong> this measures <em>direction only</em> — it does NOT
simulate the stop-losses (VWAP / opening range / 20-EMA close) or profit targets
(20–40% option gains) described in the strategies. A high direction-accuracy doesn't
guarantee the managed trade was profitable. {LOOKBACK} days of data; past results don't
guarantee the future.
</div>
</body></html>"""


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    if not API_KEY or not API_SECRET:
        sys.exit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file.")

    print(f"ORB & 20-EMA Backtest — {SYMBOL} — {LOOKBACK} days — 5-minute bars")
    print("=" * 60)

    df = fetch_bars(SYMBOL, TimeFrame(5, TimeFrameUnit.Minute), LOOKBACK, "5m")
    df = df[
        ((df.index.hour == 9) & (df.index.minute >= 30)) |
        ((df.index.hour >= 10) & (df.index.hour < 16))
    ].copy()

    print(f"  {len(df):,} bars · computing indicators…")
    df = add_indicators(df)
    df = add_outcomes(df)

    print("  detecting ORB signals…")
    orb = detect_orb(df)
    print(f"    {len(orb)} breakouts")

    print("  detecting 20-EMA pullback signals…")
    ema = detect_ema_pullback(df)
    print(f"    {len(ema)} bounces")

    if orb.empty and ema.empty:
        print("No signals found.")
        return

    print("\nBuilding report…")
    out = RESULTS_DIR / "orb_ema_report.html"
    out.write_text(build_report(orb, ema), encoding="utf-8")
    print(f"\nDone. Open: {out.resolve()}")


if __name__ == "__main__":
    main()
