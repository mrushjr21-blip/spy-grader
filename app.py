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


BEARISH_WATCH = ["AMD"]           # symbols to monitor for 3pm bearish setup
BULLISH_WATCH = ["MU", "NVDA"]   # symbols to monitor for 10-11am bullish setup

def fetch_spy_data():
    client = StockHistoricalDataClient(API_KEY, API_SECRET)
    now = datetime.now(ET)

    def fetch_et(symbol, timeframe, start):
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe,
                               start=start, end=now, feed="iex")
        return _normalize_single(client.get_stock_bars(req).df, symbol)

    bars_1m  = fetch_et("SPY", TimeFrame.Minute, now - timedelta(days=3))
    bars_5m  = fetch_et("SPY", TimeFrame(5, TimeFrameUnit.Minute), now - timedelta(days=3))
    bearish_5m = {
        sym: fetch_et(sym, TimeFrame(5, TimeFrameUnit.Minute), now - timedelta(days=3))
        for sym in BEARISH_WATCH
    }
    bullish_5m = {
        sym: fetch_et(sym, TimeFrame(5, TimeFrameUnit.Minute), now - timedelta(days=3))
        for sym in BULLISH_WATCH
    }
    return bars_1m, bars_5m, bearish_5m, bullish_5m


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
        bars_1m, bars_5m, bearish_5m, bullish_5m = fetch_spy_data()

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

        today_1m  = bars_1m[bars_1m.index.date == target_date].copy()
        today_5m  = add_indicators_5m(bars_5m[bars_5m.index.date == target_date].copy())
        morning   = score_morning_setup(today_5m)
        afternoon = score_afternoon_setup(today_5m)

        bearish_watch = {}
        for sym, df_raw in bearish_5m.items():
            sym_5m = add_indicators_5m(df_raw[df_raw.index.date == target_date].copy())
            result = score_afternoon_setup(sym_5m)
            result["price"] = round(float(sym_5m["close"].iloc[-1]), 2) if len(sym_5m) else None
            bearish_watch[sym] = result

        bullish_watch = {}
        for sym, df_raw in bullish_5m.items():
            sym_5m = add_indicators_5m(df_raw[df_raw.index.date == target_date].copy())
            result = score_morning_setup(sym_5m)
            result["price"] = round(float(sym_5m["close"].iloc[-1]), 2) if len(sym_5m) else None
            bullish_watch[sym] = result

        if afternoon["score"] >= 75 and afternoon["window_quality"] == "prime":
            overall_direction = "bearish"
        elif morning["score"] >= 75:
            overall_direction = "bullish"
        else:
            overall_direction = "neutral"

        return jsonify({
            "success": True,
            "timestamp": now_et.strftime("%Y-%m-%d %H:%M:%S ET"),
            "is_market_hours": is_market_hours,
            "is_replay": is_replay,
            "replay_date": str(target_date) if is_replay else None,
            "spy_price": round(float(today_1m["close"].iloc[-1]), 2) if len(today_1m) else None,
            "morning_setup": morning,
            "afternoon_setup": afternoon,
            "bullish_watch": bullish_watch,
            "bearish_watch": bearish_watch,
            "overall_direction": overall_direction,
            "score": afternoon["score"] if overall_direction == "bearish" else morning["score"],
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=False, host="0.0.0.0", port=port)
