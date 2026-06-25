# IK-API — Swing Trading Bot

A FastAPI-based algorithmic swing trading system connected to Interactive Brokers (IBKR).
Includes a live paper-trading bot, persistent historical data cache, multi-strategy backtesting,
and a Walk-Forward Holdout OOS Simulator.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Setup](#2-setup)
3. [Running the Server](#3-running-the-server)
4. [Generating Historical Data](#4-generating-historical-data)
5. [Running the OOS Backtest Simulator](#5-running-the-oos-backtest-simulator)
6. [Running the Live Trading Bot](#6-running-the-live-trading-bot)
7. [All API Endpoints](#7-all-api-endpoints)

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.13+ | `C:\Users\Canada\AppData\Local\Programs\Python\Python313\python.exe` |
| IBKR TWS or IB Gateway | Download from [interactivebrokers.com](https://www.interactivebrokers.com) |
| Paper trading account | Required before using real money |

### IBKR Gateway Configuration

1. Open **IB Gateway** and log in with your paper account
2. Go to **Configure → Settings → API → Settings**
3. Enable **Enable ActiveX and Socket Clients**
4. Set **Socket port** to `4002` (paper) or `4001` (live)
5. Check **Allow connections from localhost only**

---

## 2. Setup

```powershell
# Navigate to the project
cd D:\TradingApp\ik-api

# Install all dependencies
& "C:\Users\Canada\AppData\Local\Programs\Python\Python313\python.exe" -m pip install -r requirements.txt
```

**Dependencies installed:**
- `fastapi`, `uvicorn` — API server
- `ib-insync` — IBKR connection
- `matplotlib` — graphs
- `numpy`, `pandas`, `scikit-learn` — data + future ML
- `APScheduler` — 30-minute scan scheduling

---

## 3. Running the Server

```powershell
cd D:\TradingApp\ik-api

# Start the API server (hot-reload enabled)
& "C:\Users\Canada\AppData\Local\Programs\Python\Python313\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

The server starts at **http://127.0.0.1:8000**

Interactive API docs (Swagger UI): **http://127.0.0.1:8000/docs**

> **Note:** IB Gateway must be running and connected before making any IBKR data requests.
> Set `USE_MOCK=true` as an environment variable to use mock data without a live connection.

---

## 4. Generating Historical Data

There are two ways to store historical data: **manual snapshots** (for backtesting)
and the **automatic cache** (for the live bot).

---

### 4a. Manual Snapshot — Save a file to `historical_data/`

Use this to save a named snapshot for a specific duration. The file gets a
unique ID you can reference in backtest endpoints.

```http
POST http://127.0.0.1:8000/json/save_historical/TSLA
    ?duration=1 Y
    &bar_size=1 day
```

```http
POST http://127.0.0.1:8000/json/save_historical/MSFT
    ?duration=5 Y
    &bar_size=1 day
```

**Common durations:** `1 W`, `1 M`, `3 M`, `6 M`, `1 Y`, `2 Y`, `5 Y`  
**Common bar sizes:** `1 day`, `4 hours`, `1 hour`, `30 mins`, `5 mins`

The response includes the `id` (timestamp), e.g.:

```json
{
  "message": "Data saved successfully",
  "id": "1764824399773",
  "symbol": "TSLA",
  "duration": "1 Y",
  "bar_size": "1 day",
  "data_points": 250
}
```

Save this `id` — you'll use it in backtest endpoints.

**List saved files:**

```http
GET http://127.0.0.1:8000/json/list_saved_historical
```

**Load a saved file:**

```http
GET http://127.0.0.1:8000/json/load_historical/1764824399773
```

---

### 4b. Automatic Cache — Pre-warm `cache/` for the Live Bot

The bot uses a persistent cache that stores **5 years of daily** and **1 year of 4-hour**
bars per symbol. Historical bars never change, so only new bars are fetched on each update
(delta fetch). This reduces IBKR connections from ~11 per scan cycle to 2–6.

**Pre-warm cache before starting the bot:**

```http
POST http://127.0.0.1:8000/bot/warm_cache
Content-Type: application/json

{
  "symbols": ["TSLA", "MSFT", "AAPL", "SPY", "QQQ"],
  "bar_sizes": ["1 day", "4 hours"]
}
```

> The first call for each symbol fetches the full 5-year history from IBKR.
> This takes 10–30 seconds per symbol. Subsequent calls only fetch new bars.

**Check cache status:**

```http
GET http://127.0.0.1:8000/bot/cache_status
```

Response shows each file, bar count, last bar date, and file size in KB.

**Clear cache for a symbol** (forces full re-download on next scan):

```http
DELETE http://127.0.0.1:8000/bot/cache/TSLA
DELETE http://127.0.0.1:8000/bot/cache/TSLA?bar_size=4 hours
```

---

## 5. Running the OOS Backtest Simulator

The OOS simulator implements a **Walk-Forward Holdout Backtest** — the gold standard
for strategy validation in quantitative finance.

**How it works:**

```
Full dataset (e.g. 5 years of daily bars)
├─ In-Sample  (IS)  — first 60% — strategy selection + parameter optimisation
└─ Out-of-Sample (OOS) — last 40% — blind forward simulation (no fitting)
```

The algorithm optimises each strategy's parameters using IS data only (grid search, scored by
Sharpe ratio), then simulates the OOS window exactly as if trading live — it has no knowledge
of OOS prices during parameter selection.

**Strategies tested:**

| Strategy | Type |
|---|---|
| SMA Crossover | Trend following |
| RSI + Bollinger Mean Reversion | Mean reversion (highest win rate) |
| MACD Momentum | Momentum |
| BB Squeeze Breakout | Volatility breakout |
| EMA Stack (8/21/50) | Institutional trend alignment |
| N-Day High Breakout (O'Neil/CANSLIM) | Momentum + volume |
| Combined Signal Score | Multi-indicator composite |

---

### 5a. Run OOS Backtest on a Saved File

First save a **5–10 year** daily file using the steps in section 4a, then:

```http
POST http://127.0.0.1:8000/api/oos_backtest/1764824399773
    ?is_pct=0.6
    &initial_capital=10000
    &commission=1.0
    &slippage=0.0005
    &save_graph=true
```

| Parameter | Default | Description |
|---|---|---|
| `is_pct` | `0.6` | In-Sample fraction (0.6 = first 60% for training, last 40% blind) |
| `initial_capital` | `10000` | Starting capital per strategy simulation |
| `commission` | `1.0` | Flat commission per trade in dollars |
| `slippage` | `0.0005` | Slippage fraction (0.05%) |
| `save_graph` | `true` | Save equity curve PNG to `graphs/` |

**Response includes:**
- `terminology` — human-readable description of the IS/OOS split
- `data_summary` — bar counts, date ranges, OOS duration in years
- `strategies` — per-strategy IS and OOS performance metrics
- `summary_table` — formatted text comparison table (print directly)
- `graph_saved_to` — path to the saved PNG

**Print the summary table from the response:**

```python
import requests, json

resp = requests.post("http://127.0.0.1:8000/api/oos_backtest/YOUR_FILE_ID?save_graph=true")
data = resp.json()
print(data["terminology"])
print(data["summary_table"])
```

---

### 5b. Run OOS Backtest from Cache (recommended for 5Y+ data)

If you have already warmed the cache for a symbol:

```http
POST http://127.0.0.1:8000/api/oos_backtest_cached/TSLA
    ?is_pct=0.6
    &initial_capital=10000
    &save_graph=true
```

This uses `cache/tsla_1_day.json` (5 years of daily bars) automatically.

---

### 5c. Run OOS Backtest from the Command Line (no server required)

```powershell
cd D:\TradingApp\ik-api

& "C:\Users\Canada\AppData\Local\Programs\Python\Python313\python.exe" -c @"
import json, sys
sys.path.insert(0, '.')
from oos_backtest import OOSBacktest

# Load your data file
with open('historical_data/tsla_1764824399773.json') as f:
    data = json.load(f)['data']

print(f'Loaded {len(data)} bars')

# Run the backtest (60% IS, 40% OOS)
engine = OOSBacktest(data, is_pct=0.6, initial_capital=10000)
results = engine.run()

# Print the comparison table
print(results['terminology'])
print()
print(engine.format_table(results['strategies'], oos_years=results['data_summary']['oos_years']))

# Save the equity curve graph
path = engine.generate_graph(results, 'graphs/my_oos_result.png', symbol='TSLA')
print('Graph saved to:', path)
"@
```

**Reading the output table:**

```
Strategy                 |  IS Ret | IS Sharpe |  OOS Ret | OOS Ann. | OOS Sharpe |  Max DD |  Trades |   Win%
ema_stack                |  +24.6% |      1.78 |   +16.8% |   +18.2% |       1.32 |    6.1% |      25 |    62%
```

| Column | Meaning |
|---|---|
| `IS Ret` | Return during the training window (higher = strategy found good signals) |
| `IS Sharpe` | Risk-adjusted IS return — used to select best parameters |
| `OOS Ret` | **Total return in the blind simulation window** — the honest result |
| `OOS Ann.` | OOS return annualised — compare against SPY 10%/yr benchmark |
| `OOS Sharpe` | Risk-adjusted OOS return — above 1.0 is good, above 1.5 is excellent |
| `Max DD` | Worst peak-to-trough drawdown in the OOS window |
| `Trades` | Number of completed round-trips in OOS |
| `Win%` | Percentage of OOS trades that were profitable |

**Red flags (strategy likely curve-fitted):**
- IS Sharpe >> OOS Sharpe (large drop-off)
- IS return strongly positive but OOS negative
- Very few OOS trades (< 5) — insufficient sample size

> **Recommendation:** Use at least 5 years of daily data so the OOS window covers at least
> 2 full years and includes both bull and bear market conditions.

---

## 6. Running the Live Trading Bot

The bot scans your watchlist every 30 minutes on weekdays 09:30–16:00 ET,
evaluates macro + micro regime, sizes positions using ATR-based stops,
and routes paper orders through IBKR port 4002.

### Step 1 — Pre-warm the cache

```http
POST http://127.0.0.1:8000/bot/warm_cache
Content-Type: application/json

{
  "symbols": ["TSLA", "MSFT", "AAPL", "SPY"],
  "bar_sizes": ["1 day", "4 hours"]
}
```

### Step 2 — Start the bot

```http
POST http://127.0.0.1:8000/bot/start
Content-Type: application/json

{
  "watchlist": ["TSLA", "MSFT", "AAPL"],
  "initial_capital": 5000,
  "risk_pct": 0.015,
  "allow_short": false,
  "max_positions": 3,
  "atr_stop_multiplier": 2.0,
  "profit_target_rr": 2.0,
  "max_hold_days": 15
}
```

| Parameter | Default | Description |
|---|---|---|
| `watchlist` | required | Symbols to scan |
| `initial_capital` | `5000` | Starting cash |
| `risk_pct` | `0.015` | Max risk per trade (1.5% = $75 on a $5k account) |
| `allow_short` | `false` | Enable short selling during bear regimes |
| `max_positions` | `3` | Max concurrent open positions |
| `atr_stop_multiplier` | `2.0` | Stop loss = entry ± (2 × ATR) |
| `profit_target_rr` | `2.0` | Profit target = 2:1 reward:risk |
| `max_hold_days` | `15` | Force-close positions held longer than this |

### Step 3 — Monitor the bot

**Check status and open positions:**
```http
GET http://127.0.0.1:8000/bot/status
```

**View performance vs SPY 10% benchmark:**
```http
GET http://127.0.0.1:8000/bot/performance
```

**Check current regime for your symbols:**
```http
GET http://127.0.0.1:8000/bot/regime?symbols=TSLA,MSFT,AAPL
```

**Trigger an immediate scan (for testing outside market hours):**
```http
POST http://127.0.0.1:8000/bot/scan_now
Content-Type: application/json

["TSLA", "MSFT"]
```

### Step 4 — Stop the bot

```http
POST http://127.0.0.1:8000/bot/stop
```

State is persisted to `portfolio_state.json` — the bot can be restarted and
will resume tracking open positions.

### Circuit Breakers (automatic halt)

The bot automatically stops trading if:
- Daily P&L ≤ −5%
- Weekly P&L ≤ −8%
- Total drawdown from peak ≤ −15%

Check `bot_status` in `/bot/status` — it will show `"circuit_breaker"` with a reason.
To resume after reviewing, restart with `/bot/start`.

---

## 7. All API Endpoints

### Historical Data  (`/json`)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/json/save_historical/{symbol}` | Fetch from IBKR and save to `historical_data/` |
| `GET` | `/json/load_historical/{file_id}` | Load a previously saved file by ID |
| `GET` | `/json/analyze_file/{file_id}` | Basic statistics on a saved file |
| `GET` | `/json/list_saved_historical` | List all saved files with metadata |

### Backtesting  (`/api`)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/backtest/{file_id}` | Single train/validation split backtest (SMA crossover) |
| `POST` | `/api/walk_forward/{file_id}` | Walk-forward analysis across N sliding windows |
| `GET` | `/api/optimize_strategy/{file_id}` | Grid-search best SMA parameters on training data |
| `POST` | `/api/regime_backtest/{file_id}` | Backtest with macro/micro regime gating |
| `POST` | `/api/oos_backtest/{file_id}` | **Walk-Forward Holdout Backtest** on a saved file |
| `POST` | `/api/oos_backtest_cached/{symbol}` | Walk-Forward Holdout Backtest using the data cache |

### Bot Control  (`/bot`)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/bot/start` | Start the bot with watchlist and config |
| `POST` | `/bot/stop` | Stop the scheduler and persist state |
| `GET` | `/bot/status` | Current positions, P&L, and bot status |
| `GET` | `/bot/performance` | Full metrics vs 10% SPY annual benchmark |
| `GET` | `/bot/regime?symbols=...` | Macro/micro regime for each symbol |
| `POST` | `/bot/scan_now` | Trigger an immediate scan cycle |
| `POST` | `/bot/warm_cache` | Pre-download historical data for watchlist symbols |
| `GET` | `/bot/cache_status` | List all cached files with metadata |
| `DELETE` | `/bot/cache/{symbol}` | Delete cache for a symbol (forces full re-download) |

### Charts  (root)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/historical_stock/{symbol}` | OHLCV data for symbol (1M daily) |
| `GET` | `/historical_stock_graph/{symbol}` | PNG line chart of historical prices |
| `GET` | `/swing_simulation/{symbol}` | PNG chart with MA crossover buy/sell signals |
| `GET` | `/earliest_data/{symbol}` | Earliest available date for symbol in IBKR |

---

## Project Structure

```
ik-api/
├── main.py                 # FastAPI app entry point, IBKR client config
├── indicators.py           # RSI, MACD, Bollinger Bands, ATR, ADX, Volume Ratio
├── regime.py               # Macro (SPY/VIX) + micro (per-stock) regime detection
├── risk_manager.py         # ATR stops, position sizing, Kelly, circuit breakers
├── backtest.py             # BacktestEngine + all strategy methods + API endpoints
├── oos_backtest.py         # Walk-Forward Holdout Backtest framework
├── data_cache.py           # Persistent JSON cache with delta-fetch updates
├── trading_bot.py          # BotEngine — 30-min scan loop orchestrator
├── bot_router.py           # FastAPI router for bot control endpoints
├── portfolio_state.py      # Position/trade persistence (portfolio_state.json)
├── historical_json.py      # Manual save/load endpoints for historical data
├── services/
│   └── IKBRClient.py       # ib_insync wrapper with rate limiting + SPY/VIX fetch
├── historical_data/        # Manually saved OHLCV snapshots (timestamped JSON)
├── cache/                  # Auto-managed persistent data cache (symbol + bar_size)
├── graphs/                 # Generated PNG charts
├── requirements.txt
└── README.md
```

---

## Quick Start Checklist

- [ ] IB Gateway running on port `4002` (paper account)
- [ ] `pip install -r requirements.txt`
- [ ] `uvicorn main:app --reload` — server running on port 8000
- [ ] `POST /bot/warm_cache` with your watchlist (first run: downloads 5Y daily history)
- [ ] `POST /api/oos_backtest_cached/TSLA` — validate strategy on blind OOS data
- [ ] `POST /bot/start` — begin paper trading
- [ ] `GET /bot/status` — monitor positions
