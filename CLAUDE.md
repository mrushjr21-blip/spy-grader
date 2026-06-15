# CLAUDE.md

Guidance for AI assistants (and humans) working in this repository.

## What this is

**spy-grader** is a small Flask web app that grades intraday trading setups for
SPY (and a handful of watch symbols) in real time, plus a standalone backtester
that validates the setup rules against ~1 year of historical data.

The whole project is two Python files and one HTML template — there is no build
step, no framework beyond Flask, and no test suite. Keep changes proportionate
to that simplicity.

### The trading setup being graded

Both the live grader and the backtester score the same 4-condition pattern on
**5-minute bars**:

- **Bullish setup** (looked for in the 10–11am ET window):
  1. Engulfing candle whose body engulfs the previous bar's body
  2. That candle closes **above** SMA10 (crossing up — prior bar was below)
  3. MACD histogram curling up (rising) **or** already positive
  4. A **double bottom** in the prior 20 bars (2+ local lows within 0.25%)
- **Bearish setup** (looked for at 3pm ET): the mirror image — engulfing candle
  closing **below** SMA10, MACD histogram falling or negative, **double top** in
  the prior 20 bars.

Each condition is worth 25 points → a clean setup scores 100/100. The
time-of-day "quality" labels (prime / neutral / marginal / avoid) baked into the
app are derived from the backtester's historical accuracy results.

## Files

| Path | Purpose |
|------|---------|
| `app.py` | Flask app. Fetches live/recent bars from Alpaca, scores the morning (bullish) and afternoon (bearish) setups, serves the dashboard and a JSON grading API. |
| `backtest.py` | Standalone CLI backtester. Scans ~23 symbols over `LOOKBACK` days, measures setup accuracy at +15/+30/+60 min, and writes a self-contained HTML report. |
| `backtest_orb_ema.py` | Standalone CLI backtester for two **alternative** SPY strategies — Opening Range Breakout (09:30–09:45) and 20-EMA pullback. Direction-accuracy only (no stop/target simulation). Writes `backtest_results/orb_ema_report.html`. Independent of the main 4-condition setup. |
| `templates/index.html` | The entire frontend — HTML, CSS, and vanilla JS in one file. Polls `/api/grade` every 60s and renders the cards. |
| `requirements.txt` | Python deps (Flask, alpaca-py, pandas, numpy, python-dotenv, pytz, gunicorn). |
| `.env.example` | Template for the two required Alpaca credentials. |
| `.gitignore` | Ignores `.env`, `__pycache__/`, `backtest_cache/`, `backtest_results/`. |

There are no `tests/`, `backtest_cache/`, or `backtest_results/` directories in
git — the latter two are generated at runtime and git-ignored.

## Architecture & data flow

### Live app (`app.py`)
1. `fetch_spy_data()` pulls 1m + 5m SPY bars plus 5m bars for the watch lists
   (`BULLISH_WATCH = ["MU", "NVDA"]`, `BEARISH_WATCH = ["AMD"]`) from Alpaca,
   using the free **IEX feed** and a 3-day lookback.
2. All bar DataFrames are normalized to a tz-aware **US/Eastern** DatetimeIndex
   via `_normalize_single()`.
3. `add_indicators_5m()` adds SMA10 and MACD(12/26/9) columns.
4. `score_morning_setup()` / `score_afternoon_setup()` scan the last few bars for
   a signal and return a dict of score, criteria, values, and window quality.
5. `/api/grade` assembles everything (including per-symbol watch results and an
   `overall_direction`) into one JSON payload.
6. The frontend polls `/api/grade` and re-renders.

**Replay mode:** when the market is closed (or today's data isn't available yet),
`/api/grade` falls back to the most recent available trading day and sets
`is_replay: true` so the UI shows an end-of-day snapshot.

**NumpyJSONProvider:** a custom Flask JSON provider is installed so numpy scalar
types (`np.bool_`, `np.integer`, `np.floating`) serialize cleanly. Keep this in
mind if you return new numpy-typed values from the API.

### Backtester (`backtest.py`)
- `fetch_bars()` caches each symbol's bars as a pickle in `backtest_cache/` for
  12 hours to avoid re-hitting Alpaca.
- `detect_signals()` records **every** SMA10 cross along with which of the other
  conditions were met, so the report can slice any combination of filters.
- `build_report()` writes a single self-contained HTML file to
  `backtest_results/report.html`, which the app serves at `/report`.

The two files intentionally **duplicate** the indicator and pattern-detection
logic (`calc_ema`/`add_indicators`, `_has_double_bottom`/`check_double_bottom`,
etc.). They are not shared via a common module. If you change a setup rule,
**change it in both places** or the live grader and backtest will disagree.

## Setup & running

```bash
# 1. Install deps (use a virtualenv)
pip install -r requirements.txt

# 2. Provide Alpaca credentials
cp .env.example .env        # then edit .env with your own keys

# 3. Run the web app (defaults to port 5050)
python app.py               # http://localhost:5050

# Or with gunicorn (prod-style)
gunicorn app:app -b 0.0.0.0:5050

# 4. Run the backtester (writes backtest_results/report.html)
python backtest.py
```

### Environment variables
- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` — required for any data fetching.
  Loaded via `python-dotenv` from `.env`. Without them, `/api/grade` returns a
  friendly error and `backtest.py` exits.
- `PORT` — optional, overrides the default 5050 for the Flask app.

> ⚠️ **Secrets:** `.env.example` currently contains real-looking key values.
> Treat anything resembling a credential as a secret — never commit a real
> `.env` (it's git-ignored), and don't paste live keys into code, commits, or
> chat. If you believe committed keys are real, flag it to the user.

## Conventions

- **Python style:** the existing code uses aligned assignments and section-banner
  comments (`# ───`/`# ---`). Match the local style of the file you're editing
  rather than reformatting. No linter or formatter is configured.
- **Timezones:** everything trades in **US/Eastern**. Always work with tz-aware
  timestamps; use the `ET` constant.
- **Indicators:** SMA10, MACD(12/26/9). 5-minute bars are the unit of analysis;
  `DB_BARS`/`DB_BARS_5M = 20` bars = 100 minutes of double-bottom/top lookback;
  `DB_TOL = 0.0025` (0.25%) is the price tolerance for matching swing points.
- **Scoring:** 4 conditions × 25 points = 100. "Detected" means a perfect 100.
- **Frontend:** plain HTML/CSS/JS, no framework or bundler. The dark theme uses a
  consistent palette (green `#4ade80` bullish, red `#f87171` bearish, indigo
  neutral). Window-quality badge colors are duplicated across the three render
  functions — update them together.
- **Data feed:** the free Alpaca **IEX** feed (`feed="iex"`) is used everywhere.

## Gotchas

- **Keep `app.py` and `backtest.py` in sync** for any rule/indicator change.
- The window-quality `_quality_map` strings in `app.py` encode backtest findings
  (e.g. "11am — Best window"). If the backtest conclusions change, update these
  labels and the disclaimer text in `index.html`.
- Watch lists are hardcoded constants near the top of `app.py`
  (`BULLISH_WATCH`, `BEARISH_WATCH`); the backtest universe is the `SYMBOLS` list
  in `backtest.py`.
- No automated tests exist. Verify changes by running the app against live/recent
  data (or the backtester) and eyeballing the dashboard / report.

## This is not trading advice

The app and report both carry disclaimers. Historical accuracy ≠ future results.
Don't add language that overstates the reliability of these signals.
