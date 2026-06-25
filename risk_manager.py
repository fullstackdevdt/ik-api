"""
Risk management: position sizing, ATR stops, Kelly criterion, circuit breakers.
Designed for a $5 k account with 1.5 % max risk per trade ($75).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PortfolioLimits:
    max_positions: int = 3
    max_position_pct: float = 0.50       # Max 50 % of capital in one position
    max_risk_pct_per_trade: float = 0.015  # 1.5 % default risk per trade
    daily_halt_pct: float = -0.05        # Halt if daily P&L ≤ -5 %
    weekly_halt_pct: float = -0.08       # Halt if weekly P&L ≤ -8 %
    total_halt_pct: float = -0.15        # Halt if total drawdown ≤ -15 %


DEFAULT_LIMITS = PortfolioLimits()


# ---------------------------------------------------------------------------
# Position spec returned by calculate_position_size
# ---------------------------------------------------------------------------

@dataclass
class PositionSpec:
    symbol: str
    direction: str           # "long" | "short"
    entry_price: float
    stop_price: float
    profit_target: float
    shares: int
    dollar_risk: float       # Actual $ at risk (shares × stop distance)
    position_value: float    # Total $ committed (shares × entry)
    risk_pct: float          # Realised risk as % of capital


# ---------------------------------------------------------------------------
# Core sizing functions
# ---------------------------------------------------------------------------

def calculate_atr_stop(
    entry_price: float,
    atr: float,
    multiplier: float = 2.0,
    direction: str = "long",
) -> float:
    """
    ATR-based stop loss.
      Long:  stop = entry - (multiplier × ATR)
      Short: stop = entry + (multiplier × ATR)
    """
    offset = multiplier * atr
    return round(entry_price - offset if direction == "long" else entry_price + offset, 4)


def calculate_profit_target(
    entry_price: float,
    stop_price: float,
    reward_risk_ratio: float = 2.0,
    direction: str = "long",
) -> float:
    """
    Profit target at `reward_risk_ratio` × the stop distance from entry.
    Default 2 : 1.
    """
    risk = abs(entry_price - stop_price)
    if direction == "long":
        return round(entry_price + reward_risk_ratio * risk, 4)
    return round(entry_price - reward_risk_ratio * risk, 4)


def calculate_position_size(
    capital: float,
    entry_price: float,
    stop_price: float,
    profit_target: float,
    risk_pct: float = 0.015,
    size_multiplier: float = 1.0,
    symbol: str = "",
    limits: PortfolioLimits = DEFAULT_LIMITS,
) -> PositionSpec:
    """
    Calculate shares to buy/short so that hitting the stop costs exactly
    `risk_pct × capital × size_multiplier` dollars.

    Formula:
        risk_amount = capital × risk_pct × size_multiplier
        shares      = floor(risk_amount / |entry - stop|)
        capped at   max_position_pct × capital / entry_price
    """
    zero = PositionSpec(
        symbol=symbol, direction="long",
        entry_price=entry_price, stop_price=stop_price,
        profit_target=profit_target,
        shares=0, dollar_risk=0.0, position_value=0.0, risk_pct=0.0,
    )

    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0 or entry_price <= 0 or capital <= 0:
        return zero

    risk_amount = capital * risk_pct * max(0.0, min(1.0, size_multiplier))
    shares = math.floor(risk_amount / stop_distance)

    # Cap at max position %
    max_by_capital = math.floor((capital * limits.max_position_pct) / entry_price)
    shares = max(0, min(shares, max_by_capital))

    if shares == 0:
        return zero

    direction = "long" if stop_price < entry_price else "short"
    actual_risk = shares * stop_distance

    return PositionSpec(
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        stop_price=stop_price,
        profit_target=profit_target,
        shares=shares,
        dollar_risk=round(actual_risk, 2),
        position_value=round(shares * entry_price, 2),
        risk_pct=round(actual_risk / capital * 100, 3),
    )


# ---------------------------------------------------------------------------
# Kelly criterion (fractional)
# ---------------------------------------------------------------------------

def calculate_kelly_fraction(
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    kelly_fraction: float = 0.25,
) -> float:
    """
    Quarter-Kelly fraction of capital to risk.

        f* = (b·p − q) / b        where b = avg_win / avg_loss
        safe_f = f* × kelly_fraction

    Returns 0.0 if the edge is negative or inputs are invalid.
    """
    if avg_loss_pct <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    b = avg_win_pct / avg_loss_pct
    p, q = win_rate, 1.0 - win_rate
    kelly = (b * p - q) / b
    return round(max(0.0, kelly * kelly_fraction), 4)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

def check_circuit_breaker(
    daily_pnl_pct: float,
    weekly_pnl_pct: float,
    total_drawdown_pct: float,
    limits: PortfolioLimits = DEFAULT_LIMITS,
) -> Dict:
    """
    Evaluate whether any circuit-breaker threshold has been breached.

    All inputs are signed fractions (negative = loss).
    Returns {"halt": bool, "reasons": [str], ...}.
    """
    triggered = False
    reasons: List[str] = []

    if daily_pnl_pct <= limits.daily_halt_pct:
        triggered = True
        reasons.append(
            f"Daily P&L {daily_pnl_pct:.2%} ≤ halt threshold {limits.daily_halt_pct:.2%}"
        )
    if weekly_pnl_pct <= limits.weekly_halt_pct:
        triggered = True
        reasons.append(
            f"Weekly P&L {weekly_pnl_pct:.2%} ≤ halt threshold {limits.weekly_halt_pct:.2%}"
        )
    if total_drawdown_pct <= limits.total_halt_pct:
        triggered = True
        reasons.append(
            f"Total drawdown {total_drawdown_pct:.2%} ≤ halt threshold {limits.total_halt_pct:.2%}"
        )

    return {
        "halt": triggered,
        "reasons": reasons,
        "daily_pnl_pct": round(daily_pnl_pct, 4),
        "weekly_pnl_pct": round(weekly_pnl_pct, 4),
        "total_drawdown_pct": round(total_drawdown_pct, 4),
    }


def can_open_new_position(
    current_open_count: int,
    limits: PortfolioLimits = DEFAULT_LIMITS,
) -> bool:
    """True if the portfolio still has capacity for another position."""
    return current_open_count < limits.max_positions
