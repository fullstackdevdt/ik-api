"""
Persistent portfolio state: open positions, closed trades, configuration.
Stored as JSON at portfolio_state.json in the working directory.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

STATE_FILE = "portfolio_state.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Position:
    symbol: str
    direction: str           # "long" | "short"
    entry_price: float
    shares: int
    stop_price: float
    profit_target: float
    entry_date: str          # "YYYY-MM-DD"
    strategy: str
    atr_at_entry: float
    size_multiplier: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class ClosedTrade:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    shares: int
    entry_date: str
    exit_date: str
    exit_reason: str         # "stop_loss" | "profit_target" | "signal" | "time_exit"
    pnl_dollars: float
    pnl_pct: float
    strategy: str
    hold_days: int


@dataclass
class BotConfig:
    initial_capital: float = 5000.0
    risk_pct: float = 0.015
    atr_stop_multiplier: float = 2.0
    profit_target_rr: float = 2.0
    max_hold_days: int = 15
    max_positions: int = 3
    allow_short: bool = False
    watchlist: List[str] = field(default_factory=list)


@dataclass
class PortfolioState:
    config: BotConfig = field(default_factory=BotConfig)
    capital: float = 5000.0           # Available cash (not yet allocated)
    positions: Dict[str, Position] = field(default_factory=dict)
    closed_trades: List[ClosedTrade] = field(default_factory=list)
    bot_status: str = "stopped"       # "running" | "stopped" | "circuit_breaker"
    circuit_breaker_reason: str = ""
    last_scan: str = ""
    start_of_day_capital: float = 5000.0
    start_of_week_capital: float = 5000.0
    peak_capital: float = 5000.0


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def load_state(filepath: str = STATE_FILE) -> PortfolioState:
    """Load state from JSON; returns a fresh default state if file absent."""
    if not os.path.exists(filepath):
        return PortfolioState()

    with open(filepath, "r") as f:
        raw = json.load(f)

    config_data = raw.get("config", {})
    config = BotConfig(**{k: v for k, v in config_data.items() if k in BotConfig.__dataclass_fields__})

    cap = raw.get("capital", config.initial_capital)
    state = PortfolioState(
        config=config,
        capital=cap,
        bot_status=raw.get("bot_status", "stopped"),
        circuit_breaker_reason=raw.get("circuit_breaker_reason", ""),
        last_scan=raw.get("last_scan", ""),
        start_of_day_capital=raw.get("start_of_day_capital", cap),
        start_of_week_capital=raw.get("start_of_week_capital", cap),
        peak_capital=raw.get("peak_capital", cap),
    )

    for sym, pd in raw.get("positions", {}).items():
        state.positions[sym] = Position(**{k: v for k, v in pd.items() if k in Position.__dataclass_fields__})

    for td in raw.get("closed_trades", []):
        state.closed_trades.append(ClosedTrade(**{k: v for k, v in td.items() if k in ClosedTrade.__dataclass_fields__}))

    return state


def save_state(state: PortfolioState, filepath: str = STATE_FILE) -> None:
    """Persist current state to JSON."""
    raw = {
        "config": asdict(state.config),
        "capital": state.capital,
        "bot_status": state.bot_status,
        "circuit_breaker_reason": state.circuit_breaker_reason,
        "last_scan": state.last_scan,
        "start_of_day_capital": state.start_of_day_capital,
        "start_of_week_capital": state.start_of_week_capital,
        "peak_capital": state.peak_capital,
        "positions": {sym: asdict(pos) for sym, pos in state.positions.items()},
        "closed_trades": [asdict(t) for t in state.closed_trades],
    }
    with open(filepath, "w") as f:
        json.dump(raw, f, indent=2)


# ---------------------------------------------------------------------------
# Performance summary
# ---------------------------------------------------------------------------

def get_performance_summary(state: PortfolioState) -> Dict:
    """Derive performance metrics from closed trades and current capital."""
    trades = state.closed_trades

    # Mark-to-market total
    total_capital = state.capital + sum(
        p.current_price * p.shares for p in state.positions.values()
    )

    if not trades:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl_dollars": 0.0,
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "initial_capital": state.config.initial_capital,
            "current_capital": round(total_capital, 2),
            "peak_capital": round(state.peak_capital, 2),
            "exit_reasons": {},
            "open_positions": len(state.positions),
        }

    wins = [t for t in trades if t.pnl_dollars > 0]
    losses = [t for t in trades if t.pnl_dollars <= 0]
    total_pnl = sum(t.pnl_dollars for t in trades)
    win_rate = len(wins) / len(trades) * 100
    avg_win = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0.0

    # Simplified Sharpe from per-trade returns
    returns = [t.pnl_pct for t in trades]
    sharpe = 0.0
    if len(returns) >= 2:
        mean_r = sum(returns) / len(returns)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1))
        if std_r > 0:
            sharpe = round((mean_r / std_r) * math.sqrt(len(returns)), 4)

    total_return_pct = (total_capital - state.config.initial_capital) / state.config.initial_capital * 100
    drawdown_pct = (
        (state.peak_capital - total_capital) / state.peak_capital * 100
        if state.peak_capital > 0 else 0.0
    )

    annualized = 0.0
    try:
        first_dt = datetime.strptime(trades[0].entry_date, "%Y-%m-%d")
        last_dt = datetime.strptime(trades[-1].exit_date, "%Y-%m-%d")
        days = max((last_dt - first_dt).days, 1)
        annualized = ((1 + total_return_pct / 100) ** (365 / days) - 1) * 100
    except Exception:
        pass

    exit_reasons: Dict[str, int] = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 2),
        "total_pnl_dollars": round(total_pnl, 2),
        "total_return_pct": round(total_return_pct, 2),
        "annualized_return_pct": round(annualized, 2),
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": round(drawdown_pct, 2),
        "initial_capital": state.config.initial_capital,
        "current_capital": round(total_capital, 2),
        "peak_capital": round(state.peak_capital, 2),
        "exit_reasons": exit_reasons,
        "open_positions": len(state.positions),
    }
