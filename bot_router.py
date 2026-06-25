"""
FastAPI router for bot control and monitoring.
Prefix: /bot   Tags: ["Bot Control"]
"""
from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Body, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

router = APIRouter()

_bot_engine = None
_scheduler = None


def _get_engine():
    """Lazily initialise BotEngine with the shared IBKR client."""
    global _bot_engine
    if _bot_engine is None:
        from main import ib_client
        from trading_bot import BotEngine
        _bot_engine = BotEngine(ib_client)
    return _bot_engine


# ---------------------------------------------------------------------------
# Start / Stop
# ---------------------------------------------------------------------------

@router.post("/start")
async def start_bot(
    watchlist: List[str] = Body(..., description="Stock symbols to trade, e.g. [\"MSFT\",\"TSLA\"]"),
    initial_capital: float = Body(5000.0),
    risk_pct: float = Body(0.015, description="Max risk per trade as a fraction, e.g. 0.015 = 1.5%"),
    allow_short: bool = Body(False),
    max_positions: int = Body(3),
    atr_stop_multiplier: float = Body(2.0),
    profit_target_rr: float = Body(2.0, description="Reward:risk ratio for profit target"),
    max_hold_days: int = Body(15),
):
    """
    Initialise and start the swing trading bot.
    Schedules a scan cycle every 30 minutes Mon–Fri 09:30–16:00 ET.
    """
    global _bot_engine, _scheduler

    def _start():
        global _bot_engine, _scheduler

        if _scheduler and _scheduler.running:
            return {"message": "Bot already running", "status": "running"}

        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        from main import ib_client
        from portfolio_state import save_state
        from trading_bot import BotEngine

        _bot_engine = BotEngine(ib_client)
        cfg = _bot_engine.state.config
        cfg.watchlist = [s.upper() for s in watchlist]
        cfg.initial_capital = initial_capital
        cfg.risk_pct = risk_pct
        cfg.allow_short = allow_short
        cfg.max_positions = max_positions
        cfg.atr_stop_multiplier = atr_stop_multiplier
        cfg.profit_target_rr = profit_target_rr
        cfg.max_hold_days = max_hold_days

        _bot_engine.state.capital = initial_capital
        _bot_engine.state.peak_capital = initial_capital
        _bot_engine.state.start_of_day_capital = initial_capital
        _bot_engine.state.start_of_week_capital = initial_capital
        _bot_engine.state.bot_status = "running"
        _bot_engine.state.circuit_breaker_reason = ""
        save_state(_bot_engine.state)

        _scheduler = BackgroundScheduler(timezone="America/New_York")
        _scheduler.add_job(
            _bot_engine.run_cycle,
            CronTrigger(day_of_week="mon-fri", hour="9-15", minute="0,30"),
            id="swing_scan",
            replace_existing=True,
        )
        _scheduler.start()

        return {
            "message": "Bot started",
            "status": "running",
            "watchlist": cfg.watchlist,
            "config": {
                "initial_capital": initial_capital,
                "risk_pct": risk_pct,
                "allow_short": allow_short,
                "max_positions": max_positions,
                "atr_stop_multiplier": atr_stop_multiplier,
                "profit_target_rr": profit_target_rr,
                "max_hold_days": max_hold_days,
            },
        }

    result = await run_in_threadpool(_start)
    return JSONResponse(result)


@router.post("/stop")
async def stop_bot():
    """Stop the scheduler and persist current state."""
    global _scheduler

    def _stop():
        global _scheduler
        if _scheduler and _scheduler.running:
            _scheduler.shutdown(wait=False)
            _scheduler = None

        engine = _get_engine()
        engine.state.bot_status = "stopped"
        from portfolio_state import save_state
        save_state(engine.state)
        return {"message": "Bot stopped", "status": "stopped"}

    result = await run_in_threadpool(_stop)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Status & Performance
# ---------------------------------------------------------------------------

@router.get("/status")
async def bot_status():
    """Current bot status, open positions, and real-time P&L summary."""
    def _status():
        from portfolio_state import get_performance_summary, load_state
        state = load_state()

        positions_out = {
            sym: {
                "direction": p.direction,
                "shares": p.shares,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "stop_price": p.stop_price,
                "profit_target": p.profit_target,
                "entry_date": p.entry_date,
                "strategy": p.strategy,
                "unrealized_pnl": p.unrealized_pnl,
            }
            for sym, p in state.positions.items()
        }

        total = state.capital + sum(
            p.current_price * p.shares for p in state.positions.values()
        )

        return {
            "bot_status": state.bot_status,
            "circuit_breaker_reason": state.circuit_breaker_reason,
            "last_scan": state.last_scan,
            "watchlist": state.config.watchlist,
            "available_cash": round(state.capital, 2),
            "total_portfolio_value": round(total, 2),
            "open_positions": positions_out,
            "performance": get_performance_summary(state),
        }

    result = await run_in_threadpool(_status)
    return JSONResponse(result)


@router.get("/performance")
async def bot_performance():
    """Detailed performance metrics including comparison to 10% SPY benchmark."""
    def _perf():
        from portfolio_state import get_performance_summary, load_state
        state = load_state()
        perf = get_performance_summary(state)
        perf["spy_annual_benchmark_pct"] = 10.0
        perf["beats_benchmark"] = perf.get("annualized_return_pct", 0.0) > 10.0
        return perf

    result = await run_in_threadpool(_perf)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Regime & Manual scan
# ---------------------------------------------------------------------------

@router.get("/regime")
async def current_regime(
    symbols: str = Query(..., description="Comma-separated symbols, e.g. MSFT,TSLA"),
):
    """Classify the current macro/micro regime for each requested symbol."""
    def _regime():
        engine = _get_engine()

        spy_closes = []
        vix_close = None
        try:
            ctx = engine._fetch_spy_vix_context()
            spy_closes = ctx.get("spy_closes", [])
            vix_close = ctx.get("vix_close")
        except Exception as e:
            pass

        regimes = {}
        for sym in [s.strip().upper() for s in symbols.split(",") if s.strip()]:
            try:
                regimes[sym] = engine.scan_symbol(sym, spy_closes, vix_close).get("regime", {})
            except Exception as ex:
                regimes[sym] = {"error": str(ex)}

        return {
            "macro_context": {"vix": vix_close, "spy_bars": len(spy_closes)},
            "regimes": regimes,
        }

    result = await run_in_threadpool(_regime)
    return JSONResponse(result)


@router.post("/scan_now")
async def scan_now(
    symbols: Optional[List[str]] = Body(None, description="Override watchlist for this one scan"),
):
    """Trigger an immediate scan cycle (useful for testing outside market hours)."""
    def _scan():
        engine = _get_engine()
        if symbols:
            engine.state.config.watchlist = [s.upper() for s in symbols]
        return engine.run_cycle()

    result = await run_in_threadpool(_scan)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

@router.post("/warm_cache")
async def warm_cache(
    symbols: List[str] = Body(..., description="Symbols to pre-download, e.g. [\"MSFT\",\"TSLA\",\"SPY\"]"),
    bar_sizes: List[str] = Body(
        ["1 day", "4 hours"],
        description="Bar sizes to cache, e.g. [\"1 day\",\"4 hours\"]",
    ),
):
    """
    Pre-download and cache historical data for the given symbols before starting
    the bot.  First call for each symbol fetches the full initial history
    (5 Y of daily, 1 Y of 4-hour); subsequent calls only fetch the delta.

    Run this once per symbol when you add a new ticker to your watchlist.
    """
    def _warm():
        from data_cache import DataCache
        from main import ib_client
        cache = DataCache(ib_client)
        summary = cache.warm([s.upper() for s in symbols], bar_sizes)
        return {
            "status": "ok",
            "cached_bars": summary,
            "message": (
                "Historical data saved to cache/ directory. "
                "Future scans will only fetch new bars since the last cached date."
            ),
        }

    result = await run_in_threadpool(_warm)
    return JSONResponse(result)


@router.get("/cache_status")
async def cache_status():
    """List all cached files with symbol, bar size, data points, last bar date, and file size."""
    def _status():
        from data_cache import DataCache
        from main import ib_client
        cache = DataCache(ib_client)
        files = cache.status()
        total_bars = sum(f["data_points"] for f in files if f.get("data_points"))
        total_kb = sum(f["size_kb"] for f in files if f.get("size_kb"))
        return {
            "cached_files": len(files),
            "total_bars": total_bars,
            "total_size_kb": round(total_kb, 1),
            "files": files,
        }

    result = await run_in_threadpool(_status)
    return JSONResponse(result)


@router.delete("/cache/{symbol}")
async def clear_cache(
    symbol: str,
    bar_size: str = Query(None, description="Bar size to clear, e.g. '1 day'. Omit to clear all bar sizes for symbol."),
):
    """
    Delete cached file(s) for a symbol so the next scan re-downloads full history.
    Useful if IBKR data was adjusted/corrected, or to force a fresh 5Y pull.
    """
    import os
    from data_cache import DataCache, CACHE_DIR
    from main import ib_client
    cache = DataCache(ib_client)

    removed = []
    if bar_size:
        path = cache.cache_path(symbol.upper(), bar_size)
        if os.path.exists(path):
            os.remove(path)
            removed.append(os.path.basename(path))
    else:
        prefix = symbol.lower() + "_"
        for fname in os.listdir(CACHE_DIR):
            if fname.startswith(prefix) and fname.endswith(".json"):
                os.remove(os.path.join(CACHE_DIR, fname))
                removed.append(fname)

    return JSONResponse({
        "status": "ok",
        "removed": removed,
        "message": f"Next scan will re-download full history for {symbol.upper()}.",
    })
