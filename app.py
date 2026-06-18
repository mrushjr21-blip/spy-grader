from flask import Flask, render_template, jsonify
from pathlib import Path
import pandas as pd
import numpy as np
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
import pytz
import os
import traceback
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
ET = pytz.timezone("US/Eastern")

from flask.json.provider import DefaultJSONProvider

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)

API_KEY = os.environ.get("ALPACA_API_KEY", "")
API_SECRET = os.environ.get("ALPACA_SECRET_KEY", "")


# ---------------------------------------------------------------------------
# Indicator calculations
# ---------------------------------------------------------------------------

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _normalize_single(df, symbol):
    """Flatten and timezone-convert a single-symbol bar DataFrame to ET."""
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


BEARISH_WATCH      = ["NVDA", "AMD"]  # full setup bearish watch
BULLISH_WATCH      = ["MU", "NVDA"]   # full setup bullish watch
ENGULF_BULL_WATCH  = []               # 2-condition bullish watch (empty)
ENGULF_BEAR_WATCH  = ["MU"]           # 2-condition bearish watch (prime: 2pm only)

# Hardcoded backtest stats for specific symbol/direction/hour combos (full setup, 365 days)
# Keys: (symbol, direction, ET_hour)
BACKTEST_STATS = {
    # MU full setup bullish (prime: 10-11am)
    ("MU",   "bullish", 10): {"wr_15m": 75, "wr_30m": 75, "wr_60m": 75, "avg_15m": "+0.41%", "avg_30m": "+0.48%", "avg_60m": "+0.98%", "mfe": "+1.31%", "note": "Best exit: hold full hour (+60m)"},
    ("MU",   "bullish", 11): {"wr_15m": 75, "wr_30m": 65, "wr_60m": 75, "avg_15m": "+0.12%", "avg_30m": "+0.14%", "avg_60m": "+0.12%", "mfe": "+0.54%", "note": "Similar edge at +15m & +60m"},
    # NVDA full setup bullish (prime: 10-11am)
    ("NVDA", "bullish", 10): {"wr_15m": 58, "wr_30m": 69, "wr_60m": 62, "avg_15m": "-0.03%", "avg_30m": "+0.09%", "avg_60m": "+0.04%", "mfe": "+0.55%", "note": "Best exit: +30m (69% win rate)"},
    ("NVDA", "bullish", 11): {"wr_15m": 65, "wr_30m": 62, "wr_60m": 59, "avg_15m": "+0.01%", "avg_30m": "+0.04%", "avg_60m": "+0.09%", "mfe": "+0.43%", "note": "Best edge at +15m (65% win rate)"},
    # AMD full setup bearish (prime: 9:30-11am)
    ("AMD",  "bearish",  9): {"wr_15m": 54, "wr_30m": 69, "wr_60m": 54, "avg_15m": "+0.00%", "avg_30m": "+0.32%", "avg_60m": "+0.70%", "mfe": "+1.28%", "note": "Best exit: +30m (69% win rate)"},
    ("AMD",  "bearish", 10): {"wr_15m": 59, "wr_30m": 76, "wr_60m": 65, "avg_15m": "+0.09%", "avg_30m": "+0.23%", "avg_60m": "+0.19%", "mfe": "+1.09%", "note": "Best exit: +30m (76% win rate)"},
    ("AMD",  "bearish", 11): {"wr_15m": 60, "wr_30m": 55, "wr_60m": 60, "avg_15m": "+0.12%", "avg_30m": "+0.12%", "avg_60m": "+0.11%", "mfe": "+0.66%", "note": "Similar edge at +15m & +60m"},
    # NVDA full setup bearish (prime: 3pm)
    ("NVDA", "bearish", 15): {"wr_15m": 50, "wr_30m": 50, "wr_60m": 55, "avg_15m": "+0.00%", "avg_30m": "+0.08%", "avg_60m": "+0.13%", "mfe": "+0.38%", "note": "Best exit: hold full hour (+60m)"},
    # MU 2-condition bearish (prime: 2pm)
    ("MU",   "engulf_bear", 14): {"wr_15m": 57, "wr_30m": 61, "wr_60m": 50, "avg_15m": "+0.05%", "avg_30m": "+0.16%", "avg_60m": "+0.02%", "mfe": "+0.73%", "note": "Best exit: +30m (61% win rate)"},
}

def fetch_spy_data():
    client = StockHistoricalDataClient(API_KEY, API_SECRET)
    now = datetime.now(ET)

    def fetch_et(symbol, timeframe, start):
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe,
                               start=start, end=now, feed="iex")
        return _normalize_single(client.get_stock_bars(req).df, symbol)

    # SPY 1m bars used only for date detection / is_replay logic
    bars_1m = fetch_et("SPY", TimeFrame.Minute, now - timedelta(days=3))

    all_bear_syms = sorted(set(BEARISH_WATCH) | set(ENGULF_BEAR_WATCH))
    all_bull_syms = sorted(set(BULLISH_WATCH) | set(ENGULF_BULL_WATCH))

    bearish_5m = {
        sym: fetch_et(sym, TimeFrame(5, TimeFrameUnit.Minute), now - timedelta(days=3))
        for sym in all_bear_syms
    }
    bullish_5m = {
        sym: fetch_et(sym, TimeFrame(5, TimeFrameUnit.Minute), now - timedelta(days=3))
        for sym in all_bull_syms
    }
    return bars_1m, bearish_5m, bullish_5m


def add_indicators_5m(df):
    df = df.copy()
    df["sma10"] = df["close"].rolling(10, min_periods=1).mean()
    ema12 = calc_ema(df["close"], 12)
    ema26 = calc_ema(df["close"], 26)
    df["macd"]      = ema12 - ema26
    df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]
    return df


def _find_local_lows(lows, swing=2):
    idx = []
    for i in range(swing, len(lows) - swing):
        if all(lows[i] <= lows[i-j] for j in range(1, swing+1)) and \
           all(lows[i] <= lows[i+j] for j in range(1, swing+1)):
            idx.append(i)
    return idx


def _has_double_bottom(window_lows, tol=0.0025):
    arr  = list(window_lows)
    idxs = _find_local_lows(arr, swing=2)
    if len(idxs) < 2:
        return False, float("nan")
    vals = [arr[i] for i in idxs]
    for a in range(len(vals)):
        for b in range(a + 1, len(vals)):
            avg = (vals[a] + vals[b]) / 2
            if abs(vals[a] - vals[b]) / avg <= tol:
                return True, min(vals[a], vals[b])
    return False, float("nan")


def _find_local_highs(highs, swing=2):
    idx = []
    for i in range(swing, len(highs) - swing):
        if all(highs[i] >= highs[i-j] for j in range(1, swing+1)) and \
           all(highs[i] >= highs[i+j] for j in range(1, swing+1)):
            idx.append(i)
    return idx


def _has_double_top(window_highs, tol=0.0025):
    arr  = list(window_highs)
    idxs = _find_local_highs(arr, swing=2)
    if len(idxs) < 2:
        return False, float("nan")
    vals = [arr[i] for i in idxs]
    for a in range(len(vals)):
        for b in range(a + 1, len(vals)):
            avg = (vals[a] + vals[b]) / 2
            if abs(vals[a] - vals[b]) / avg <= tol:
                return True, max(vals[a], vals[b])
    return False, float("nan")


# ---------------------------------------------------------------------------
# Pattern: SMA + Engulfing (2-condition, 76% bull / 70% bear win rate)
# ---------------------------------------------------------------------------

def score_sma_engulf(df, direction="bullish"):
    is_bull = direction == "bullish"
    result = {
        "name": "SMA + Engulfing",
        "detected": False,
        "direction": direction,
        "score": 0,
        "max_score": 100,
        "in_window": False,
        "signal_time": None,
        "bars_ago": None,
        "criteria": [],
        "values": {},
        "window_quality": "outside",
        "window_label": "Outside market hours",
    }

    if len(df) < 5:
        return result

    last     = df.iloc[-1]
    now_hour = last.name.hour
    now_min  = last.name.minute

    if is_bull:
        _qmap = {
            9:  ("context",  "9:30am — Pre-window, watch only"),
            10: ("prime",    "10am — Prime window (76% win rate)"),
            11: ("prime",    "11am — Prime window (76% win rate)"),
            12: ("neutral",  "12pm — Outside prime window"),
            13: ("marginal", "1pm — Outside prime window"),
            14: ("avoid",    "2pm — Avoid bullish"),
            15: ("avoid",    "3pm — Avoid bullish"),
        }
        wq, wlabel = _qmap.get(now_hour, ("outside", "Outside market hours"))
        result["in_window"] = now_hour in (10, 11)
    else:
        if now_hour == 15:
            wq, wlabel = "prime", "3pm — Prime window (70% win rate)"
        elif now_hour == 14 and now_min >= 45:
            wq, wlabel = "context", "2:45pm — Pre-window, watch only"
        else:
            _qmap = {
                9:  ("context", "9:30am — Watch only"),
                10: ("neutral", "10am — Outside prime window"),
                11: ("avoid",   "11am — Avoid bearish"),
                12: ("avoid",   "12pm — Avoid bearish"),
                13: ("avoid",   "1pm — Avoid bearish"),
                14: ("avoid",   "2pm — Avoid bearish"),
            }
            wq, wlabel = _qmap.get(now_hour, ("outside", "Outside market hours"))
        result["in_window"] = now_hour == 15

    result["window_quality"] = wq
    result["window_label"]   = wlabel

    found = None
    for bars_ago in range(0, min(5, len(df) - 2)):
        i    = len(df) - 1 - bars_ago
        bar  = df.iloc[i]
        prev = df.iloc[i - 1]

        c, o   = float(bar["close"]),  float(bar["open"])
        pc, po = float(prev["close"]), float(prev["open"])
        s, ps  = float(bar["sma10"]),  float(prev["sma10"])

        body_lo,  body_hi  = min(o, c),   max(o, c)
        pbody_lo, pbody_hi = min(po, pc), max(po, pc)

        if is_bull:
            sma_cross = (pc < ps) and (c > s)
            engulf    = (c > o) and (body_lo <= pbody_lo) and (body_hi >= pbody_hi)
        else:
            sma_cross = (pc > ps) and (c < s)
            engulf    = (c < o) and (body_lo <= pbody_lo) and (body_hi >= pbody_hi)

        if not sma_cross:
            continue

        score = 50 + (50 if engulf else 0)
        if found is None or score > found["score"]:
            found = {
                "bars_ago": bars_ago,
                "engulf":   engulf,
                "score":    score,
                "bar_time": bar.name.strftime("%H:%M"),
                "sma10":    s,
                "price":    c,
            }

    def crit(label, passed, points):
        result["criteria"].append({
            "label": label, "pass": bool(passed),
            "points": points, "earned": points if passed else 0,
        })

    cross_label  = "above" if is_bull else "below"
    cross_from   = "from below" if is_bull else "from above"
    engulf_label = "Bullish engulfing candle body (engulfs previous)" if is_bull else "Bearish engulfing candle body (engulfs previous)"

    if found:
        crit(f"Engulfing candle closes {cross_label} SMA10 ({cross_from})", True, 50)
        crit(engulf_label, found["engulf"], 50)
        result["score"]       = found["score"]
        result["detected"]    = found["score"] == 100
        result["signal_time"] = found["bar_time"]
        result["bars_ago"]    = found["bars_ago"]
        result["values"] = {
            "Signal Bar": found["bar_time"] + ("  ← current bar" if found["bars_ago"] == 0 else f"  ({found['bars_ago']*5}min ago)"),
            "SMA10 (5m)": f"${found['sma10']:.2f}",
            "Price":      f"${found['price']:.2f}",
        }
    else:
        last_sma   = float(df.iloc[-1]["sma10"])
        last_close = float(df.iloc[-1]["close"])
        crit(f"Engulfing candle closes {cross_label} SMA10 ({cross_from})", False, 50)
        crit(engulf_label, False, 50)
        result["values"] = {
            "Price vs SMA10": f"${last_close - last_sma:+.2f}",
            "SMA10 (5m)":     f"${last_sma:.2f}",
            "No SMA10 cross": "in last 20 min",
        }

    return result


# ---------------------------------------------------------------------------
# Pattern: 10–11am bullish setup (5m bars)
# ---------------------------------------------------------------------------

DB_BARS_5M = 20   # 100 minutes of lookback on 5m chart

def score_morning_setup(df):
    result = {
        "name": "10–11am Bullish Setup (5m)",
        "detected": False,
        "direction": "bullish",
        "score": 0,
        "max_score": 100,
        "in_window": False,
        "signal_time": None,
        "bars_ago": None,
        "vol_warning": False,
        "rel_vol": None,
        "criteria": [],
        "values": {},
    }

    # Compute window quality from whatever bars we have before the early return
    if len(df) > 0:
        _h = df.iloc[-1].name.hour
        _quality_map = {
            9:  ("context",  "9:30am — Pre-window, watch only"),
            10: ("prime",    "10am — Strong at +60m (61%)"),
            11: ("prime",    "11am — Best window (62%/68%/62%)"),
            12: ("neutral",  "12pm — Marginal, degrades quickly"),
            13: ("marginal", "1pm — Slight edge at +15m only (59%)"),
            14: ("avoid",    "2pm — Below coin flip, avoid"),
            15: ("avoid",    "3pm — Never trade bullish here (0% at +60m)"),
        }
        wq, wlabel = _quality_map.get(_h, ("outside", "Outside market hours"))
        result["window_quality"] = wq
        result["window_label"]   = wlabel
        result["in_window"]      = _h in (10, 11)

    if len(df) < DB_BARS_5M + 3:
        return result

    last = df.iloc[-1]
    now_hour = last.name.hour
    result["in_window"] = now_hour in (10, 11)

    # Backtest-derived time-of-day quality for bullish signals
    _quality_map = {
        9:  ("context",  "9:30am — Pre-window, watch only"),
        10: ("prime",    "10am — Strong at +60m (61%)"),
        11: ("prime",    "11am — Best window (62%/68%/62%)"),
        12: ("neutral",  "12pm — Marginal, degrades quickly"),
        13: ("marginal", "1pm — Slight edge at +15m only (59%)"),
        14: ("avoid",    "2pm — Below coin flip, avoid"),
        15: ("avoid",    "3pm — Never trade bullish here (0% at +60m)"),
    }
    wq, wlabel = _quality_map.get(now_hour, ("outside", "Outside market hours"))
    result["window_quality"] = wq
    result["window_label"]   = wlabel

    # Search the last 4 bars (20 min) for a valid signal
    found = None
    for bars_ago in range(0, min(5, len(df) - DB_BARS_5M - 1)):
        i   = len(df) - 1 - bars_ago
        bar  = df.iloc[i]
        prev = df.iloc[i - 1]

        c, o   = float(bar["close"]),  float(bar["open"])
        pc, po = float(prev["close"]), float(prev["open"])
        s, ps  = float(bar["sma10"]),  float(prev["sma10"])
        h, ph  = float(bar["macd_hist"]), float(prev["macd_hist"])

        sma_cross = (pc < ps) and (c > s)
        if not sma_cross:
            continue

        body_lo,  body_hi  = min(o, c),   max(o, c)
        pbody_lo, pbody_hi = min(po, pc), max(po, pc)
        engulf = (c > o) and (body_lo <= pbody_lo) and (body_hi >= pbody_hi)

        macd_curl = (h > ph) or (h > 0)   # curling up toward cross OR already crossed bullish

        window_lows = df["low"].values[max(0, i - DB_BARS_5M): i]
        db_found, support = _has_double_bottom(window_lows)

        score = 25 + (25 if engulf else 0) + (25 if macd_curl else 0) + (25 if db_found else 0)

        avg_vol = df["volume"].mean() if "volume" in df.columns else 1
        rel_vol = float(bar["volume"]) / avg_vol if avg_vol > 0 and "volume" in df.columns else 1.0

        if found is None or score > found["score"]:
            found = {
                "bars_ago": bars_ago,
                "engulf":   engulf,
                "macd_curl": macd_curl,
                "db_found": db_found,
                "support":  support,
                "score":    score,
                "bar_time": bar.name.strftime("%H:%M"),
                "sma10":    s,
                "macd_hist": h,
                "price":    c,
                "rel_vol":  rel_vol,
            }

    def crit(label, passed, points):
        result["criteria"].append({
            "label": label,
            "pass":  bool(passed),
            "points": points,
            "earned": points if passed else 0,
        })

    if found:
        f = found
        crit("Engulfing candle body (current engulfs previous)", f["engulf"],    25)
        crit("Engulfing candle closes above SMA10 (cross from below)", True,     25)
        crit("MACD histogram curling up or already crossed bullish",    f["macd_curl"], 25)
        crit(f"Double bottom in prior {DB_BARS_5M} bars (≤0.25% apart)", f["db_found"], 25)

        result["score"]       = f["score"]
        result["detected"]    = f["score"] == 100
        result["signal_time"] = f["bar_time"]
        result["bars_ago"]    = f["bars_ago"]
        result["rel_vol"]     = round(f["rel_vol"], 2)
        result["vol_warning"] = f["rel_vol"] < 0.8
        result["values"] = {
            "Signal Bar":    f["bar_time"] + ("  ← current bar" if f["bars_ago"] == 0 else f"  ({f['bars_ago']*5}min ago)"),
            "SMA10 (5m)":   f"${f['sma10']:.2f}",
            "Support Level": f"${f['support']:.2f}" if not (f["support"] != f["support"]) else "—",
            "MACD Hist":    f"{f['macd_hist']:+.4f}",
            "Signal Vol":   f"{f['rel_vol']:.2f}× avg",
        }
    else:
        last_sma  = float(df.iloc[-1]["sma10"])
        last_hist = float(df.iloc[-1]["macd_hist"])
        last_close = float(df.iloc[-1]["close"])
        crit("Engulfing candle body (current engulfs previous)", False, 25)
        crit("Engulfing candle closes above SMA10 (cross from below)", False, 25)
        crit("MACD histogram curling up", last_hist > float(df.iloc[-2]["macd_hist"]), 25)
        crit(f"Double bottom in prior {DB_BARS_5M} bars (≤0.25% apart)", False, 25)
        result["score"] = sum(c["earned"] for c in result["criteria"])
        result["values"] = {
            "Price vs SMA10": f"${last_close - last_sma:+.2f}",
            "SMA10 (5m)":     f"${last_sma:.2f}",
            "MACD Hist":      f"{last_hist:+.4f}",
            "No SMA10 cross": "in last 20 min",
        }

    return result


# ---------------------------------------------------------------------------
# Pattern: 3pm bearish setup (5m bars)
# ---------------------------------------------------------------------------

def score_afternoon_setup(df):
    result = {
        "name": "3pm Bearish Setup (5m)",
        "detected": False,
        "direction": "bearish",
        "score": 0,
        "max_score": 100,
        "in_window": False,
        "signal_time": None,
        "bars_ago": None,
        "vol_warning": False,
        "rel_vol": None,
        "criteria": [],
        "values": {},
    }

    # Compute window quality from whatever bars we have before the early return
    if len(df) > 0:
        _h = df.iloc[-1].name.hour
        _m = df.iloc[-1].name.minute
        if _h == 15:
            wq, wlabel = "prime",   "3pm — Strongest bearish window (100% at +60m)"
        elif _h == 14 and _m >= 45:
            wq, wlabel = "context", "2:45pm — Pre-window, watch only"
        else:
            _quality_map = {
                9:  ("context", "9:30am — Watch only"),
                10: ("neutral", "10am — Mixed bearish results"),
                11: ("avoid",   "11am — Weak bearish, avoid"),
                12: ("avoid",   "12pm — Weak bearish, avoid"),
                13: ("avoid",   "1pm — Weak bearish, avoid"),
                14: ("avoid",   "2pm — Weak bearish, avoid"),
            }
            wq, wlabel = _quality_map.get(_h, ("outside", "Outside market hours"))
        result["window_quality"] = wq
        result["window_label"]   = wlabel
        result["in_window"]      = _h == 15

    if len(df) < DB_BARS_5M + 3:
        return result

    last = df.iloc[-1]
    now_hour   = last.name.hour
    now_minute = last.name.minute
    result["in_window"] = now_hour == 15

    if now_hour == 15:
        wq, wlabel = "prime",   "3pm — Strongest bearish window (100% at +60m)"
    elif now_hour == 14 and now_minute >= 45:
        wq, wlabel = "context", "2:45pm — Pre-window, watch only"
    else:
        _quality_map = {
            9:  ("context", "9:30am — Watch only"),
            10: ("neutral", "10am — Mixed bearish results"),
            11: ("avoid",   "11am — Weak bearish, avoid"),
            12: ("avoid",   "12pm — Weak bearish, avoid"),
            13: ("avoid",   "1pm — Weak bearish, avoid"),
            14: ("avoid",   "2pm — Weak bearish, avoid"),
        }
        wq, wlabel = _quality_map.get(now_hour, ("outside", "Outside market hours"))
    result["window_quality"] = wq
    result["window_label"]   = wlabel

    found = None
    for bars_ago in range(0, min(5, len(df) - DB_BARS_5M - 1)):
        i    = len(df) - 1 - bars_ago
        bar  = df.iloc[i]
        prev = df.iloc[i - 1]

        c, o   = float(bar["close"]),  float(bar["open"])
        pc, po = float(prev["close"]), float(prev["open"])
        s, ps  = float(bar["sma10"]),  float(prev["sma10"])
        h, ph  = float(bar["macd_hist"]), float(prev["macd_hist"])

        sma_cross = (pc > ps) and (c < s)   # was above SMA10, now below
        if not sma_cross:
            continue

        body_lo,  body_hi  = min(o, c),   max(o, c)
        pbody_lo, pbody_hi = min(po, pc), max(po, pc)
        engulf = (c < o) and (body_lo <= pbody_lo) and (body_hi >= pbody_hi)

        macd_curl = (h < ph) or (h < 0)   # falling toward cross OR already crossed bearish

        window_highs = df["high"].values[max(0, i - DB_BARS_5M): i]
        dt_found, resistance = _has_double_top(window_highs)

        score = 25 + (25 if engulf else 0) + (25 if macd_curl else 0) + (25 if dt_found else 0)

        avg_vol = df["volume"].mean() if "volume" in df.columns else 1
        rel_vol = float(bar["volume"]) / avg_vol if avg_vol > 0 and "volume" in df.columns else 1.0

        if found is None or score > found["score"]:
            found = {
                "bars_ago":   bars_ago,
                "engulf":     engulf,
                "macd_curl":  macd_curl,
                "dt_found":   dt_found,
                "resistance": resistance,
                "score":      score,
                "bar_time":   bar.name.strftime("%H:%M"),
                "sma10":      s,
                "macd_hist":  h,
                "price":      c,
                "rel_vol":    rel_vol,
            }

    def crit(label, passed, points):
        result["criteria"].append({
            "label": label,
            "pass":  bool(passed),
            "points": points,
            "earned": points if passed else 0,
        })

    if found:
        f = found
        crit("Engulfing candle body (current engulfs previous)", f["engulf"],    25)
        crit("Engulfing candle closes below SMA10 (cross from above)", True,     25)
        crit("MACD histogram falling or already crossed bearish",  f["macd_curl"], 25)
        crit(f"Double top in prior {DB_BARS_5M} bars (≤0.25% apart)", f["dt_found"], 25)

        result["score"]       = f["score"]
        result["detected"]    = f["score"] == 100
        result["signal_time"] = f["bar_time"]
        result["bars_ago"]    = f["bars_ago"]
        result["rel_vol"]     = round(f["rel_vol"], 2)
        result["vol_warning"] = f["rel_vol"] < 0.8
        result["values"] = {
            "Signal Bar":   f["bar_time"] + ("  ← current bar" if f["bars_ago"] == 0 else f"  ({f['bars_ago']*5}min ago)"),
            "SMA10 (5m)":  f"${f['sma10']:.2f}",
            "Resistance":  f"${f['resistance']:.2f}" if not (f["resistance"] != f["resistance"]) else "—",
            "MACD Hist":   f"{f['macd_hist']:+.4f}",
            "Signal Vol":  f"{f['rel_vol']:.2f}× avg",
        }
    else:
        last_sma   = float(df.iloc[-1]["sma10"])
        last_hist  = float(df.iloc[-1]["macd_hist"])
        last_close = float(df.iloc[-1]["close"])
        crit("Engulfing candle body (current engulfs previous)", False, 25)
        crit("Engulfing candle closes below SMA10 (cross from above)", False, 25)
        crit("MACD histogram falling", last_hist < float(df.iloc[-2]["macd_hist"]), 25)
        crit(f"Double top in prior {DB_BARS_5M} bars (≤0.25% apart)", False, 25)
        result["score"] = sum(c["earned"] for c in result["criteria"])
        result["values"] = {
            "Price vs SMA10": f"${last_close - last_sma:+.2f}",
            "SMA10 (5m)":     f"${last_sma:.2f}",
            "MACD Hist":      f"{last_hist:+.4f}",
            "No SMA10 cross": "in last 20 min",
        }

    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/report")
def report():
    path = Path("backtest_results/report.html")
    if not path.exists():
        return "No report found. Run backtest.py first.", 404
    return path.read_text(encoding="utf-8")


@app.route("/api/grade")
def grade():
    if not API_KEY or not API_SECRET:
        return jsonify({
            "success": False,
            "error": "Alpaca API keys not set. Copy .env.example to .env and fill in your keys.",
        })

    try:
        bars_1m, bearish_5m, bullish_5m = fetch_spy_data()

        now_et = datetime.now(ET)
        mo = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        mc = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        is_market_hours = mo <= now_et <= mc and now_et.weekday() < 5

        today = now_et.date()
        available_dates = sorted(set(bars_1m.index.date))

        if not available_dates:
            return jsonify({"success": False, "error": "No bar data available from Alpaca."})

        if today in available_dates and is_market_hours:
            target_date = today
            is_replay = False
        else:
            target_date = available_dates[-1]
            is_replay = True

        engulf_bear_watch = {}
        for sym, df_raw in bearish_5m.items():
            if sym not in ENGULF_BEAR_WATCH:
                continue
            sym_5m = add_indicators_5m(df_raw[df_raw.index.date == target_date].copy())
            result = score_sma_engulf(sym_5m, "bearish")
            if sym == "MU":
                mu_hour = now_et.hour
                result["in_window"] = mu_hour == 14
                if mu_hour == 14:
                    result["window_quality"] = "prime"
                    result["window_label"]   = "2pm — MU bearish prime window"
                else:
                    result["window_quality"] = "avoid"
                    result["window_label"]   = "Outside MU bearish window (prime: 2pm)"
                    result["suppressed"]     = True
                    result["detected"]       = False
                    result["score"]          = 0
                    result["criteria"]       = []
                    result["values"]         = {}
            result["price"] = round(float(sym_5m["close"].iloc[-1]), 2) if len(sym_5m) else None
            engulf_bear_watch[sym] = result

        bearish_watch = {}
        for sym, df_raw in bearish_5m.items():
            if sym not in BEARISH_WATCH:
                continue
            sym_5m = add_indicators_5m(df_raw[df_raw.index.date == target_date].copy())
            result = score_afternoon_setup(sym_5m)
            if sym == "AMD":
                amd_hour = now_et.hour
                result["in_window"] = amd_hour in (9, 10, 11)
                if amd_hour in (9, 10, 11):
                    result["window_quality"] = "prime"
                    result["window_label"]   = "9:30–11am — AMD bearish prime window"
                else:
                    result["window_quality"] = "avoid"
                    result["window_label"]   = "Outside AMD bearish window (prime: 9:30–11am)"
                    result["suppressed"]     = True
                    result["detected"]       = False
                    result["score"]          = 0
                    result["criteria"]       = []
                    result["values"]         = {}
            result["price"] = round(float(sym_5m["close"].iloc[-1]), 2) if len(sym_5m) else None
            bearish_watch[sym] = result

        bullish_watch = {}
        for sym, df_raw in bullish_5m.items():
            sym_5m = add_indicators_5m(df_raw[df_raw.index.date == target_date].copy())
            result = score_morning_setup(sym_5m)
            if sym in ("MU", "NVDA"):
                bull_hour = now_et.hour
                result["in_window"] = bull_hour in (10, 11)
                if bull_hour in (10, 11):
                    result["window_quality"] = "prime"
                    result["window_label"]   = f"10–11am — {sym} bullish prime window"
                else:
                    result["window_quality"] = "avoid"
                    result["window_label"]   = f"Outside {sym} bullish window (prime: 10–11am)"
                    result["suppressed"]     = True
                    result["detected"]       = False
                    result["score"]          = 0
                    result["criteria"]       = []
                    result["values"]         = {}
            result["price"] = round(float(sym_5m["close"].iloc[-1]), 2) if len(sym_5m) else None
            bullish_watch[sym] = result

        # Inject backtest stats for the current hour into watch cards
        now_hour = now_et.hour
        for sym, d in bullish_watch.items():
            s = BACKTEST_STATS.get((sym, "bullish", now_hour))
            if s:
                d["backtest_stats"] = s
        for sym, d in bearish_watch.items():
            s = BACKTEST_STATS.get((sym, "bearish", now_hour))
            if s:
                d["backtest_stats"] = s
        for sym, d in engulf_bear_watch.items():
            s = BACKTEST_STATS.get((sym, "engulf_bear", now_hour))
            if s:
                d["backtest_stats"] = s

        return jsonify({
            "success": True,
            "timestamp": now_et.strftime("%Y-%m-%d %H:%M:%S ET"),
            "is_market_hours": is_market_hours,
            "is_replay": is_replay,
            "replay_date": str(target_date) if is_replay else None,
            "bullish_watch": bullish_watch,
            "bearish_watch": bearish_watch,
            "engulf_bear_watch": engulf_bear_watch,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=False, host="0.0.0.0", port=port)
