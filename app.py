from flask import Flask, render_template, jsonify, request
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
import threading
import urllib.request
import json as _json
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

API_KEY           = os.environ.get("ALPACA_API_KEY", "")
API_SECRET        = os.environ.get("ALPACA_SECRET_KEY", "")
DISCORD_WEBHOOK   = os.environ.get("DISCORD_WEBHOOK_URL", "")

# ─── Alert deduplication ─────────────────────────────────────────────────────
# Tracks (sym, direction, signal_time, date) tuples already fired today.
# Resets automatically because signal_time + date makes keys day-unique.
_alerted = set()


def _send_discord(payload: dict):
    """Send Discord webhook POST synchronously so errors appear in logs."""
    if not DISCORD_WEBHOOK:
        print("[discord] no webhook URL set")
        return
    try:
        body = _json.dumps(payload).encode()
        req  = urllib.request.Request(
            DISCORD_WEBHOOK,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=10)
        print(f"[discord] status={resp.status}")
    except Exception as exc:
        print(f"[discord] error: {exc}")


def _maybe_alert(sym, result, direction_label, now_et):
    """
    Send a Discord alert if this is a new prime-window signal not yet fired today.
    Fires on full setup (100) or 3-of-4 (75) during the prime window.
    """
    if not DISCORD_WEBHOOK:
        return
    wq    = result.get("window_quality", "")
    score = result.get("score", 0)
    sig_t = result.get("signal_time")
    price = result.get("price")

    if wq != "prime" or score < 75 or not sig_t:
        return

    key = (sym, direction_label, sig_t, now_et.date())
    if key in _alerted:
        return
    _alerted.add(key)

    is_full    = score == 100
    is_bull    = direction_label == "bullish"
    color      = 0x4ade80 if is_bull else 0xf87171   # green / red
    emoji      = "🟢" if is_bull else "🔴"
    label      = "FULL SETUP" if is_full else "3 OF 4"
    opt_lean   = "BUY CALLS" if is_bull else "BUY PUTS"
    bars_ago   = result.get("bars_ago", 0)
    ago_str    = f" ({bars_ago * 5}min ago)" if bars_ago else ""
    price_str  = f"${price:.2f}" if price else "—"
    window_lbl = result.get("window_label", "")

    # Criteria summary
    crit_lines = "\n".join(
        f"{'✅' if c['pass'] else '❌'} {c['label']}"
        for c in result.get("criteria", [])
    )

    # Backtest stats if available
    bs      = result.get("backtest_stats")
    bt_text = ""
    if bs:
        bt_text = (
            f"\n**Backtest (365d):** "
            f"+15m {bs['wr_15m']}% · +30m {bs['wr_30m']}% · +60m {bs['wr_60m']}%"
            f"\n_{bs['note']}_"
        )

    _send_discord({
        "embeds": [{
            "title":       f"{emoji} {sym} — {label}",
            "description": (
                f"**{opt_lean}** · Score: **{score}/100** · {sig_t}{ago_str}\n"
                f"Price: **{price_str}** · Window: {window_lbl}\n\n"
                f"{crit_lines}"
                f"{bt_text}"
            ),
            "color": color,
            "footer": {"text": f"SPY Setup Grader · {now_et.strftime('%Y-%m-%d %H:%M ET')}"},
        }]
    })


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


BEARISH_WATCH      = ["NVDA", "AMD", "MU"]                    # full setup bearish watch
BULLISH_WATCH      = ["MU", "NVDA", "NFLX", "AMAT", "HOOD"]  # full setup bullish watch
ENGULF_BULL_WATCH  = []                     # 2-condition bullish watch (empty)
ENGULF_BEAR_WATCH  = []                     # 2-condition bearish watch (empty)
GAP_FILL_TICKERS   = ["SPY", "QQQ", "IWM"] # morning gap fill + VWAP pullback
PDL_SHORT_TICKERS  = ["MSFT", "GOOGL"]         # prior day low short setup

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
    # NFLX full setup bullish (prime: 9:30am)
    ("NFLX", "bullish",  9): {"wr_15m": 64, "wr_30m": 71, "wr_60m": 64, "avg_15m": "+0.12%", "avg_30m": "+0.23%", "avg_60m": "+0.33%", "mfe": "+0.81%", "note": "Best exit: +30m (71% win rate)"},
    # AMAT full setup bullish (prime: 11am)
    ("AMAT", "bullish", 11): {"wr_15m": 65, "wr_30m": 64, "wr_60m": 61, "avg_15m": "+0.10%", "avg_30m": "+0.12%", "avg_60m": "+0.16%", "mfe": "+0.44%", "note": "Best exit: +15m (65% win rate)"},
    # HOOD full setup bullish (prime: 10am)
    ("HOOD", "bullish", 10): {"wr_15m": 55, "wr_30m": 63, "wr_60m": 59, "avg_15m": "+0.05%", "avg_30m": "+0.33%", "avg_60m": "+0.32%", "mfe": "+1.00%", "note": "Best exit: +30m (63% win rate)"},
    # NVDA full setup bearish (prime: 1-2pm)
    ("NVDA", "bearish", 13): {"wr_15m": 63, "wr_30m": 68, "wr_60m": 55, "avg_15m": "+0.04%", "avg_30m": "+0.04%", "avg_60m": "+0.02%", "mfe": "+0.38%", "note": "Best exit: +30m (68% win rate)"},
    ("NVDA", "bearish", 14): {"wr_15m": 64, "wr_30m": 55, "wr_60m": 45, "avg_15m": "+0.06%", "avg_30m": "+0.06%", "avg_60m": "-0.08%", "mfe": "+0.34%", "note": "Best exit: +15m (64% win rate)"},
    # MU full setup bearish (prime: 2pm)
    ("MU",   "bearish", 14): {"wr_15m": 65, "wr_30m": 75, "wr_60m": 50, "avg_15m": "+0.13%", "avg_30m": "+0.28%", "avg_60m": "+0.17%", "mfe": "+0.70%", "note": "Best exit: +30m (75% win rate)"},
}

# ---------------------------------------------------------------------------
# Pattern: Morning Gap Fill (SPY / QQQ / IWM)
# ---------------------------------------------------------------------------

GAP_FILL_STATS = {
    "SPY": {"wr_15m": 82, "wr_30m": 77, "n": 22},
    "QQQ": {"wr_15m": 78, "wr_30m": 72, "n": 18},
    "IWM": {"wr_15m": 78, "wr_30m": 70, "n": 27},
}

PDL_SHORT_STATS = {
    "MSFT":  {"wr_30m": 80, "n": 20, "note": "Entry before 10am · exit +30m"},
    "GOOGL": {"wr_30m": 85, "n": 13, "note": "Open >1% above PDL · exit +30m"},
}
GOOGL_MIN_DIST = 0.01  # GOOGL needs today's open 1%+ above prior day low

# ---------------------------------------------------------------------------
# Gap + Pivot Zone Detection (daily cache — zones don't change intraday)
# ---------------------------------------------------------------------------

_GP_CACHE: dict = {}   # (ticker, date) -> result dict

def compute_gap_pivot_zones(ticker: str) -> dict:
    """Return today's active gap+pivot confluence zones for ticker."""
    PIVOT_N    = 5
    CONFLUENCE = 0.75
    MIN_GAP    = 0.10
    WARMUP     = 210

    today     = datetime.now(ET).date()
    cache_key = (ticker, today)
    if cache_key in _GP_CACHE:
        return _GP_CACHE[cache_key]

    blank = dict(ticker=ticker, price=None, sma50=None, above_sma50=None, zones=[], error=None)
    if not API_KEY or not API_SECRET:
        return blank

    try:
        client = StockHistoricalDataClient(API_KEY, API_SECRET)
        req    = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=datetime.now(ET) - timedelta(days=3 * 365 + 30),
            end=datetime.now(ET),
            feed="iex",
        )
        daily = _normalize_single(client.get_stock_bars(req).df, ticker)
        daily = daily[["open", "high", "low", "close"]].dropna().copy()
    except Exception as exc:
        blank["error"] = str(exc)
        return blank

    daily["sma50"] = daily["close"].rolling(50).mean()

    gaps = []
    for i in range(1, len(daily)):
        ph = float(daily["high"].iloc[i - 1])
        o  = float(daily["open"].iloc[i])
        if o > ph and (o - ph) / ph * 100 >= MIN_GAP:
            gaps.append(dict(idx=i, bottom=ph, top=o, mid=(ph + o) / 2))

    piv_lows = []
    for i in range(PIVOT_N, len(daily) - PIVOT_N):
        w = slice(i - PIVOT_N, i + PIVOT_N + 1)
        if float(daily["low"].iloc[i]) == float(daily["low"].iloc[w].min()):
            piv_lows.append(dict(idx=i, price=float(daily["low"].iloc[i])))

    if len(daily) < WARMUP:
        _GP_CACHE[cache_key] = blank
        return blank

    last    = len(daily) - 1
    close   = float(daily["close"].iloc[last])
    sma50   = float(daily["sma50"].iloc[last])

    sup_gaps  = [g for g in gaps  if g["idx"] < last and g["top"] <= close * 1.01]
    past_pivs = [p for p in piv_lows if p["idx"] < last - PIVOT_N]

    zones = []
    for gap in reversed(sup_gaps[-20:]):
        for piv in reversed(past_pivs[-30:]):
            if abs(piv["price"] - gap["mid"]) / gap["mid"] * 100 <= CONFLUENCE:
                zones.append(dict(
                    zone_top=round(gap["top"], 2),
                    zone_bottom=round(min(gap["bottom"], piv["price"]), 2),
                    dist_pct=round((close - gap["top"]) / close * 100, 2),
                ))
                break

    result = dict(
        ticker=ticker,
        price=round(close, 2),
        sma50=round(sma50, 2),
        above_sma50=bool(close > sma50),
        zones=zones[:6],
        error=None,
    )
    _GP_CACHE[cache_key] = result
    return result


def score_gap_pivot_live(ticker: str, df_5m_today: pd.DataFrame,
                          gp_data: dict, is_post_market: bool = False) -> dict:
    """
    Live intraday step-tracking for gap+pivot setup.
    Statuses: no_zones | pre_market | no_zone_nearby | watching_touch |
              signal | in_trade | win | loss | eod_exit | expired | after_close
    """
    FIXED_TP   = 2.25
    FIXED_STOP = 1.75
    CUTOFF_T   = pd.Timestamp("15:00").time()

    zones       = gp_data.get("zones", [])
    above_sma50 = gp_data.get("above_sma50")

    result = dict(
        ticker      = ticker,
        status      = "no_zones",
        zones       = zones,
        above_sma50 = above_sma50,
        price       = gp_data.get("price"),
        sma50       = gp_data.get("sma50"),
        day_open    = None,
        active_zone = None,
        touch_time  = None,
        entry       = None,
        tp          = None,
        sl          = None,
        entry_time  = None,
        exit_price  = None,
        pnl         = None,
    )

    if not zones:
        return result

    has_bars   = df_5m_today is not None and len(df_5m_today) > 0
    last_bar   = df_5m_today.iloc[-1] if has_bars else None
    last_time  = last_bar.name if last_bar is not None else None

    # Pre-market or no bars yet
    is_pre = (not has_bars) or (
        last_time.hour < 9 or (last_time.hour == 9 and last_time.minute < 30)
    )
    if is_pre:
        result["status"] = "pre_market"
        if last_bar is not None:
            result["price"] = round(float(last_bar["close"]), 2)
        return result

    day_bars = df_5m_today.between_time("09:30", "15:55")
    if len(day_bars) == 0:
        result["status"] = "pre_market"
        return result

    day_open = float(day_bars["open"].iloc[0])
    cur_close = float(day_bars["close"].iloc[-1])
    result["day_open"] = round(day_open, 2)
    result["price"]    = round(cur_close, 2)

    # Which zone is today's open above (within 0.5% tolerance)?
    candidates = [z for z in zones if z["zone_top"] <= day_open * 1.005]
    if not candidates:
        result["status"] = "no_zone_nearby"
        return result

    zone = max(candidates, key=lambda z: z["zone_top"])
    zt   = zone["zone_top"]
    result["active_zone"] = zone

    # Scan for first zone touch
    touch_k = None
    for k in range(len(day_bars)):
        if day_bars.iloc[k].name.time() >= CUTOFF_T:
            break
        if float(day_bars["low"].iloc[k]) <= zt * 1.002:
            touch_k = k
            break

    if touch_k is None:
        current_t = day_bars.iloc[-1].name.time()
        if is_post_market or current_t >= CUTOFF_T:
            result["status"] = "expired"
        else:
            result["status"] = "watching_touch"
        return result

    result["touch_time"] = day_bars.iloc[touch_k].name.strftime("%H:%M")

    if touch_k >= len(day_bars) - 1:
        result["status"] = "signal"
        return result

    # Entry bar
    entry_bar = day_bars.iloc[touch_k + 1]
    entry     = float(entry_bar["open"])
    tp        = round(entry + FIXED_TP,   2)
    sl        = round(entry - FIXED_STOP, 2)
    result.update(entry=round(entry, 2), tp=tp, sl=sl,
                  entry_time=entry_bar.name.strftime("%H:%M"))

    # Simulate TP/stop through remaining bars
    remaining  = day_bars.iloc[touch_k + 1:]
    outcome    = None
    exit_price = None

    for k in range(len(remaining)):
        bar = remaining.iloc[k]
        if float(bar["low"]) <= sl:
            outcome, exit_price = "loss", sl
            break
        if float(bar["high"]) >= tp:
            outcome, exit_price = "win", tp
            break

    if outcome is None:
        if is_post_market:
            outcome, exit_price = "eod_exit", cur_close
        else:
            outcome, exit_price = "in_trade", cur_close

    result.update(
        status     = outcome,
        exit_price = round(exit_price, 2) if exit_price is not None else None,
        pnl        = round(exit_price - entry, 2) if exit_price is not None else None,
    )
    return result


def score_gap_fill(ticker, df_5m_today, prior_close, is_post_market=False):
    """
    Gap fill signal: 0.3–0.5% gap at open, fade with opening drive filter.
    Status: no_gap | small | large | pre_market | after_close | watching | skip | signal | expired
    """
    result = {
        "ticker":          ticker,
        "status":          "no_gap",
        "direction":       None,
        "gap_pct":         None,
        "entry":           None,
        "stop":            None,
        "target":          None,
        "prior_close":     round(prior_close, 2) if prior_close else None,
        "backtest_stats":  GAP_FILL_STATS.get(ticker),
        "pre_market":      False,
        "pre_market_time": None,
    }

    if prior_close is None or len(df_5m_today) < 1:
        return result

    # Pre-market: all bars before 9:30am — use latest bar close as projected open
    last_bar = df_5m_today.iloc[-1]
    last_time = last_bar.name
    if last_time.hour < 9 or (last_time.hour == 9 and last_time.minute < 30):
        current_price = float(last_bar["close"])
        gap           = current_price - prior_close
        gap_pct       = abs(gap) / prior_close
        direction     = "short" if gap > 0 else "long"
        result.update({
            "pre_market":      True,
            "pre_market_time": last_time.strftime("%H:%M"),
            "direction":       direction,
            "gap_pct":         round(gap_pct * 100, 3),
            "entry":           round(current_price, 2),
            "target":          round(prior_close, 2),
        })
        if gap_pct < 0.001:
            result["status"] = "no_gap"
        elif gap_pct < 0.003:
            result["status"] = "small"
        elif gap_pct > 0.005:
            result["status"] = "large"
        else:
            result["status"] = "pre_market"
            if 0.003 <= gap_pct < 0.004:
                result["bucket_note"] = "0.3-0.4% bucket — strongest at +15m (67% win rate)"
            elif 0.004 <= gap_pct < 0.005:
                result["bucket_note"] = "0.4-0.5% bucket — better at +30m (69% win rate)"
        return result

    # Post-market: 4pm+ — reset card and show tomorrow's prior close reference
    if is_post_market:
        today_close = float(df_5m_today.iloc[-1]["close"])
        result.update({
            "status":      "after_close",
            "today_close": round(today_close, 2),
            "gap_short_min": round(today_close * 1.003, 2),  # gap up 0.3%
            "gap_short_max": round(today_close * 1.005, 2),  # gap up 0.5%
            "gap_long_min":  round(today_close * 0.995, 2),  # gap down 0.5%
            "gap_long_max":  round(today_close * 0.997, 2),  # gap down 0.3%
        })
        return result

    # Market hours logic
    bar0       = df_5m_today.iloc[0]
    today_open = float(bar0["open"])
    gap        = today_open - prior_close
    gap_pct    = abs(gap) / prior_close
    direction  = "short" if gap > 0 else "long"

    result["gap_pct"]   = round(gap_pct * 100, 3)
    result["direction"] = direction
    result["entry"]     = round(today_open, 2)
    result["target"]    = round(prior_close, 2)
    if 0.003 <= gap_pct < 0.004:
        result["bucket_note"] = "0.3-0.4% bucket — strongest at +15m (67% win rate)"
    elif 0.004 <= gap_pct < 0.005:
        result["bucket_note"] = "0.4-0.5% bucket — better at +30m (69% win rate)"

    if gap_pct < 0.001:
        result["status"] = "no_gap"
        return result
    if gap_pct < 0.003:
        result["status"] = "small"
        return result
    if gap_pct > 0.005:
        result["status"] = "large"
        return result

    stop_dist    = 1.5 * abs(gap)
    result["stop"] = round(
        today_open + stop_dist if direction == "short" else today_open - stop_dist, 2
    )

    # Need first bar to close before checking opening drive filter
    if len(df_5m_today) < 2:
        result["status"] = "watching"
        return result

    bar0_close = float(df_5m_today.iloc[0]["close"])
    bar0_open  = float(df_5m_today.iloc[0]["open"])
    if direction == "short" and bar0_close > bar0_open:
        result["status"] = "skip"   # gap up + first bar closes up = continuation
        return result
    if direction == "long" and bar0_close < bar0_open:
        result["status"] = "skip"   # gap down + first bar closes down = continuation
        return result

    # Expired after 11 AM
    if df_5m_today.iloc[-1].name.hour >= 11:
        result["status"] = "expired"
        return result

    result["status"] = "signal"
    return result


# ---------------------------------------------------------------------------
# Pattern: Prior Day Low Short (MSFT 80% / GOOGL 85% at +30m)
# ---------------------------------------------------------------------------

def score_pdl_short(ticker, df_5m_today, prior_day_row, is_post_market=False):
    """
    PDL short: stock gapped up, opened above PDL, first close below PDL before 10am.
    MSFT: 80% win at +30m (n=20). GOOGL: 85% win at +30m, needs open >1% above PDL (n=13).
    Status: no_signal | pre_market | wrong_dir | already_below | dist_too_small |
            watching | signal | expired | after_close
    """
    result = {
        "ticker":         ticker,
        "status":         "no_signal",
        "pdl":            None,
        "today_open":     None,
        "dist_pct":       None,
        "gap_pct":        None,
        "entry":          None,
        "stop":           None,
        "signal_time":    None,
        "backtest_stats": PDL_SHORT_STATS.get(ticker),
    }

    if prior_day_row is None or len(df_5m_today) < 1:
        return result

    pdl      = float(prior_day_row["low"])
    pd_close = float(prior_day_row["close"])
    result["pdl"] = round(pdl, 2)

    # Post-market: show today's PDL as reference for tomorrow
    if is_post_market:
        today_close = float(df_5m_today.iloc[-1]["close"])
        result.update({
            "status":      "after_close",
            "today_close": round(today_close, 2),
        })
        return result

    # Pre-market: project direction and distance from PDL
    last_bar  = df_5m_today.iloc[-1]
    last_time = last_bar.name
    if last_time.hour < 9 or (last_time.hour == 9 and last_time.minute < 30):
        current_price = float(last_bar["close"])
        gap_pct  = (current_price - pd_close) / pd_close
        dist_pct = (current_price - pdl) / pdl
        result.update({
            "today_open":     round(current_price, 2),
            "gap_pct":        round(gap_pct * 100, 3),
            "dist_pct":       round(dist_pct * 100, 3),
            "pre_market_time": last_time.strftime("%H:%M"),
        })
        if gap_pct <= 0:
            result["status"] = "wrong_dir"
        elif dist_pct < 0:
            result["status"] = "already_below"
        elif ticker == "GOOGL" and dist_pct < GOOGL_MIN_DIST:
            result["status"] = "dist_too_small"
        else:
            result["status"] = "pre_market"
        return result

    # Market hours
    bar0       = df_5m_today.iloc[0]
    today_open = float(bar0["open"])
    gap_pct    = (today_open - pd_close) / pd_close
    dist_pct   = (today_open - pdl) / pdl

    result["today_open"] = round(today_open, 2)
    result["gap_pct"]    = round(gap_pct * 100, 3)
    result["dist_pct"]   = round(dist_pct * 100, 3)

    if gap_pct <= 0:
        result["status"] = "wrong_dir"
        return result

    if today_open <= pdl:
        result["status"] = "already_below"
        return result

    if ticker == "GOOGL" and dist_pct < GOOGL_MIN_DIST:
        result["status"] = "dist_too_small"
        return result

    # Scan for first bar that closes below PDL before 10am
    for _, bar in df_5m_today.iterrows():
        if bar.name.hour >= 10:
            break
        if float(bar["close"]) < pdl:
            result.update({
                "status":      "signal",
                "entry":       round(float(bar["close"]), 2),
                "stop":        round(pdl, 2),
                "signal_time": bar.name.strftime("%H:%M"),
            })
            return result

    current_time = df_5m_today.iloc[-1].name
    result["status"] = "watching" if current_time.hour < 10 else "expired"
    return result


# ---------------------------------------------------------------------------
# Pattern: VWAP First Pullback (SPY only)
# ---------------------------------------------------------------------------

def score_vwap_pullback(df_5m_today):
    """
    VWAP first pullback on SPY 5m.
    Check trend at 30-min mark (bar 6); wait for first touch of VWAP.
    Shorts only have real edge per backtest (WR 68%, PF 2.47).
    Status: waiting | no_trend | watching | signal | expired
    """
    TREND_BARS = 6
    MIN_DEV    = 0.003
    MAX_DEV    = 0.005
    VWAP_TOL   = 0.001

    result = {
        "status":    "waiting",
        "direction": None,
        "dev_pct":   None,
        "vwap":      None,
        "entry":     None,
        "stop":      None,
        "target":    None,
    }

    if len(df_5m_today) < 2:
        return result

    df = df_5m_today.copy()
    df["typical"]    = (df["high"] + df["low"] + df["close"]) / 3
    df["cum_tp_vol"] = (df["typical"] * df["volume"]).cumsum()
    df["cum_vol"]    = df["volume"].cumsum()
    df["vwap"]       = df["cum_tp_vol"] / df["cum_vol"].replace(0, np.nan)
    df["tr"]         = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(14, min_periods=1).mean()

    last     = df.iloc[-1]
    last_vwap = round(float(last["vwap"]), 2)
    result["vwap"] = last_vwap

    if last.name.hour >= 12:
        result["status"] = "expired"
        return result

    if len(df) <= TREND_BARS:
        result["status"] = "waiting"
        return result

    trend_bar   = df.iloc[TREND_BARS]
    trend_vwap  = float(trend_bar["vwap"])
    trend_close = float(trend_bar["close"])
    dev         = (trend_close - trend_vwap) / trend_vwap

    result["dev_pct"]   = round(abs(dev) * 100, 2)
    result["direction"] = "short" if dev < 0 else "long"

    # Longs have no backtested edge — only trade the short side
    if result["direction"] == "long":
        result["status"] = "no_trend"
        return result

    if abs(dev) < MIN_DEV or abs(dev) > MAX_DEV:
        result["status"] = "no_trend"
        return result

    atr = float(df["atr"].iloc[-1])

    # Scan for first VWAP touch after the 30-min trend mark
    touched     = False
    entry_price = None
    direction   = result["direction"]
    for i in range(TREND_BARS + 1, len(df)):
        bar        = df.iloc[i]
        lo, hi     = float(bar["low"]), float(bar["high"])
        vwap_level = float(bar["vwap"])
        if direction == "short" and hi >= vwap_level * (1 - VWAP_TOL):
            touched     = True
            entry_price = round(vwap_level, 2)
            break
        elif direction == "long" and lo <= vwap_level * (1 + VWAP_TOL):
            touched     = True
            entry_price = round(vwap_level, 2)
            break

    if not touched:
        result["status"] = "watching"
        result["entry"]  = last_vwap
        return result

    result["status"] = "signal"
    result["entry"]  = entry_price
    if direction == "short":
        result["stop"]   = round(entry_price + atr, 2)
        result["target"] = round(entry_price - 2 * atr, 2)
    else:
        result["stop"]   = round(entry_price - atr, 2)
        result["target"] = round(entry_price + 2 * atr, 2)

    return result


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
    gap_fill_5m = {
        sym: fetch_et(sym, TimeFrame(5, TimeFrameUnit.Minute), now - timedelta(days=3))
        for sym in GAP_FILL_TICKERS
    }
    gap_fill_1d = {
        sym: fetch_et(sym, TimeFrame(1, TimeFrameUnit.Day), now - timedelta(days=7))
        for sym in GAP_FILL_TICKERS
    }
    pdl_short_5m = {
        sym: fetch_et(sym, TimeFrame(5, TimeFrameUnit.Minute), now - timedelta(days=3))
        for sym in PDL_SHORT_TICKERS
    }
    pdl_short_1d = {
        sym: fetch_et(sym, TimeFrame(1, TimeFrameUnit.Day), now - timedelta(days=7))
        for sym in PDL_SHORT_TICKERS
    }
    return bars_1m, bearish_5m, bullish_5m, gap_fill_5m, gap_fill_1d, pdl_short_5m, pdl_short_1d


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
    return render_template("index.html", discord_webhook=DISCORD_WEBHOOK)


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
        bars_1m, bearish_5m, bullish_5m, gap_fill_5m, gap_fill_1d, pdl_short_5m, pdl_short_1d = fetch_spy_data()

        now_et = datetime.now(ET)
        mo = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        mc = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        is_market_hours = mo <= now_et <= mc and now_et.weekday() < 5
        is_pre_market = (
            now_et.weekday() < 5 and
            now_et.hour >= 8 and
            (now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 30))
        )
        is_post_market = now_et.weekday() < 5 and now_et.hour >= 16

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

        # Gap fill uses today during pre/post-market so projected/reset state shows live
        gap_fill_date = today if (is_market_hours or is_pre_market or is_post_market) else target_date

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
            if sym == "NVDA":
                nvda_bear_hour = now_et.hour
                result["in_window"] = nvda_bear_hour in (13, 14)
                if nvda_bear_hour == 13:
                    result["window_quality"] = "prime"
                    result["window_label"]   = "1pm — NVDA bearish prime window"
                elif nvda_bear_hour == 14:
                    result["window_quality"] = "prime"
                    result["window_label"]   = "2pm — NVDA bearish prime window"
                else:
                    result["window_quality"] = "avoid"
                    result["window_label"]   = "Outside NVDA bearish window (prime: 1–2pm)"
                    result["suppressed"]     = True
                    result["prime_label"]    = "1–2pm ET"
                    result["detected"]       = False
                    result["score"]          = 0
                    result["criteria"]       = []
                    result["values"]         = {}
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
                    result["prime_label"]    = "9:30–11am ET"
                    result["detected"]       = False
                    result["score"]          = 0
                    result["criteria"]       = []
                    result["values"]         = {}
            if sym == "MU":
                mu_bear_hour = now_et.hour
                result["in_window"] = mu_bear_hour == 14
                if mu_bear_hour == 14:
                    result["window_quality"] = "prime"
                    result["window_label"]   = "2pm — MU bearish prime window"
                else:
                    result["window_quality"] = "avoid"
                    result["window_label"]   = "Outside MU bearish window (prime: 2pm)"
                    result["suppressed"]     = True
                    result["prime_label"]    = "2pm ET"
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
                    result["prime_label"]    = "10–11am ET"
                    result["detected"]       = False
                    result["score"]          = 0
                    result["criteria"]       = []
                    result["values"]         = {}
            if sym == "NFLX":
                bull_hour = now_et.hour
                result["in_window"] = bull_hour == 9
                if bull_hour == 9:
                    result["window_quality"] = "prime"
                    result["window_label"]   = "9:30am — NFLX bullish prime window"
                else:
                    result["window_quality"] = "avoid"
                    result["window_label"]   = "Outside NFLX bullish window (prime: 9:30am)"
                    result["suppressed"]     = True
                    result["prime_label"]    = "9:30am ET"
                    result["detected"]       = False
                    result["score"]          = 0
                    result["criteria"]       = []
                    result["values"]         = {}
            if sym == "AMAT":
                bull_hour = now_et.hour
                result["in_window"] = bull_hour == 11
                if bull_hour == 11:
                    result["window_quality"] = "prime"
                    result["window_label"]   = "11am — AMAT bullish prime window"
                else:
                    result["window_quality"] = "avoid"
                    result["window_label"]   = "Outside AMAT bullish window (prime: 11am)"
                    result["suppressed"]     = True
                    result["prime_label"]    = "11am ET"
                    result["detected"]       = False
                    result["score"]          = 0
                    result["criteria"]       = []
                    result["values"]         = {}
            if sym == "HOOD":
                bull_hour = now_et.hour
                result["in_window"] = bull_hour == 10
                if bull_hour == 10:
                    result["window_quality"] = "prime"
                    result["window_label"]   = "10am — HOOD bullish prime window"
                else:
                    result["window_quality"] = "avoid"
                    result["window_label"]   = "Outside HOOD bullish window (prime: 10am)"
                    result["suppressed"]     = True
                    result["prime_label"]    = "10am ET"
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

        # Discord alerts — only fire during live market hours, not replay
        if not is_replay and is_market_hours:
            for sym, d in bullish_watch.items():
                if not d.get("suppressed"):
                    _maybe_alert(sym, d, "bullish", now_et)
            for sym, d in bearish_watch.items():
                if not d.get("suppressed"):
                    _maybe_alert(sym, d, "bearish", now_et)

        # Gap Fill and VWAP Pullback signals
        gap_fill = {}
        for sym, df_raw in gap_fill_5m.items():
            today_5m    = df_raw[df_raw.index.date == gap_fill_date].copy()
            daily_df    = gap_fill_1d.get(sym, pd.DataFrame())
            prev_days   = daily_df[daily_df.index.date < gap_fill_date] if len(daily_df) > 0 else pd.DataFrame()
            prior_close = float(prev_days.iloc[-1]["close"]) if len(prev_days) > 0 else None
            gap_fill[sym] = score_gap_fill(sym, today_5m, prior_close, is_post_market=is_post_market)

        pdl_short = {}
        for sym, df_raw in pdl_short_5m.items():
            today_5m  = df_raw[df_raw.index.date == gap_fill_date].copy()
            daily_df  = pdl_short_1d.get(sym, pd.DataFrame())
            prev_days = daily_df[daily_df.index.date < gap_fill_date] if len(daily_df) > 0 else pd.DataFrame()
            prior_day_row = prev_days.iloc[-1] if len(prev_days) > 0 else None
            pdl_short[sym] = score_pdl_short(sym, today_5m, prior_day_row, is_post_market=is_post_market)

        spy_today = gap_fill_5m.get("SPY", pd.DataFrame())
        spy_today = spy_today[spy_today.index.date == target_date].copy() if len(spy_today) > 0 else pd.DataFrame()
        vwap_pullback = score_vwap_pullback(spy_today)

        gap_pivot = {sym: compute_gap_pivot_zones(sym) for sym in ("SPY", "QQQ")}

        # ?as_of=HH:MM truncates bars for testing (e.g. as_of=09:31 simulates just after open)
        as_of_str = request.args.get("as_of")
        as_of_time = pd.Timestamp(as_of_str).time() if as_of_str else None

        gap_pivot_live = {}
        for sym in ("SPY", "QQQ"):
            df_5m    = gap_fill_5m.get(sym, pd.DataFrame())
            today_5m = df_5m[df_5m.index.date == gap_fill_date].copy() if len(df_5m) > 0 else pd.DataFrame()
            if as_of_time is not None and len(today_5m) > 0:
                today_5m = today_5m[today_5m.index.time <= as_of_time]
            gap_pivot_live[sym] = score_gap_pivot_live(sym, today_5m, gap_pivot[sym], is_post_market)

        return jsonify({
            "success": True,
            "timestamp": now_et.strftime("%Y-%m-%d %H:%M:%S ET"),
            "is_market_hours": is_market_hours,
            "is_pre_market": is_pre_market,
            "is_replay": is_replay,
            "replay_date": str(target_date) if is_replay else None,
            "bullish_watch": bullish_watch,
            "bearish_watch": bearish_watch,
            "engulf_bear_watch": engulf_bear_watch,
            "gap_fill": gap_fill,
            "pdl_short": pdl_short,
            "vwap_pullback": vwap_pullback,
            "gap_pivot": gap_pivot,
            "gap_pivot_live": gap_pivot_live,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


@app.route("/test-alert")
def test_alert_page():
    """Browser-based test page — fires a Discord alert from the client side."""
    if not DISCORD_WEBHOOK:
        return "DISCORD_WEBHOOK_URL not set in environment.", 400
    return render_template("test_alert.html", discord_webhook=DISCORD_WEBHOOK)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=False, host="0.0.0.0", port=port)
