"""
Out-of-Sample (OOS) Backtesting Framework
==========================================
Official terminology: "Walk-Forward Holdout Backtest"
  - In-Sample  (IS):  training window — strategy selection + parameter optimisation
  - Out-of-Sample (OOS): holdout window — blind forward simulation (no fitting)

Why this matters
----------------
Any strategy that looks good on all available data may simply be curve-fitted.
Splitting IS/OOS enforces honest performance attribution:
  - IS metrics answer "could we have found this strategy?"
  - OOS metrics answer "would it have worked in the future?"

Usage (standalone)
------------------
    from oos_backtest import OOSBacktest
    engine = OOSBacktest(data, is_pct=0.6, initial_capital=10_000)
    results = engine.run()
    engine.generate_graph(results, "graphs/oos_result.png")
    print(engine.format_table(results))

Usage (API endpoint)
--------------------
    POST /api/oos_backtest/{file_id}
    POST /api/oos_backtest_cached/{symbol}
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")           # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from indicators import (
    compute_atr,
    compute_ema,
    compute_rsi,
    compute_sma,
    compute_bollinger,
    compute_volume_ratio,
    compute_all,
)
from backtest import BacktestEngine   # reuse calculate_performance + existing strategies


# =========================================================================
# Extra swing-trading strategies (proven, higher win-rate)
# =========================================================================

def bb_squeeze_breakout_strategy(
    data: List[Dict],
    squeeze_pct: int = 20,
    bandwidth_period: int = 50,
) -> List[Dict]:
    """
    Bollinger Band Squeeze Breakout  (John Bollinger)
    -------------------------------------------------
    Setup:  BB bandwidth compresses to a multi-period low (market coiling).
    Entry:  Price breaks *above* upper band on the expansion bar.
    Exit:   Price drops back below the middle band (pct_b < 0.45).

    Win rate: typically 55-65 % in trending markets.
    """
    if len(data) < bandwidth_period + 30:
        return []

    closes = [d["close"] for d in data]
    bb = compute_bollinger(closes)

    # Pre-build bandwidth array
    bandwidths: List[Optional[float]] = []
    for i in range(len(data)):
        u, m, lo = bb["upper"][i], bb["middle"][i], bb["lower"][i]
        if u and m and lo and m > 0:
            bandwidths.append((u - lo) / m * 100)
        else:
            bandwidths.append(None)

    trades: List[Dict] = []
    position = False

    for i in range(bandwidth_period, len(data)):
        bw = bandwidths[i]
        bw_prev = bandwidths[i - 1]
        pct_b = bb["pct_b"][i]
        if bw is None or bw_prev is None or pct_b is None:
            continue

        # Rolling squeeze threshold (N-th percentile of recent bandwidth)
        recent = [b for b in bandwidths[i - bandwidth_period : i] if b is not None]
        if len(recent) < bandwidth_period // 2:
            continue
        threshold = sorted(recent)[max(0, int(len(recent) * squeeze_pct / 100) - 1)]

        was_squeezing = bw_prev <= threshold
        now_expanding = bw > threshold

        if not position:
            if was_squeezing and now_expanding and pct_b >= 1.0:
                trades.append({
                    "date": data[i]["date"], "action": "BUY",
                    "price": data[i]["close"],
                    "bb_pct_b": round(pct_b, 4), "bandwidth": round(bw, 4),
                })
                position = True
        else:
            if pct_b < 0.45:
                trades.append({
                    "date": data[i]["date"], "action": "SELL",
                    "price": data[i]["close"],
                    "bb_pct_b": round(pct_b, 4),
                })
                position = False

    return trades


def ema_stack_strategy(
    data: List[Dict],
    ema_short: int = 8,
    ema_mid: int = 21,
    ema_long: int = 50,
    rsi_max: float = 72.0,
) -> List[Dict]:
    """
    EMA Stack / Trend Alignment  (Mark Minervini style)
    ---------------------------------------------------
    Setup:  EMA(short) > EMA(mid) > EMA(long) — all EMAs stacked bullishly.
    Entry:  Short EMA crosses above mid EMA (stack just aligned), RSI in 45-rsi_max.
    Exit:   Short EMA drops below mid EMA (stack breaks) OR RSI > rsi_max.

    Win rate: typically 60-70 % in trending bull markets.
    """
    closes = [d["close"] for d in data]
    ema_s = compute_ema(closes, ema_short)
    ema_m = compute_ema(closes, ema_mid)
    ema_l = compute_ema(closes, ema_long)
    rsi = compute_rsi(closes)

    trades: List[Dict] = []
    position = False

    for i in range(1, len(data)):
        es, em, el, r = ema_s[i], ema_m[i], ema_l[i], rsi[i]
        es_p, em_p = ema_s[i - 1], ema_m[i - 1]
        if any(v is None for v in (es, em, el, r, es_p, em_p)):
            continue

        aligned = es > em > el
        rsi_ok = 45.0 <= r <= rsi_max
        just_crossed = es_p <= em_p and es > em   # crossover this bar

        if not position and aligned and rsi_ok and just_crossed:
            trades.append({
                "date": data[i]["date"], "action": "BUY",
                "price": data[i]["close"], "rsi": round(r, 2),
            })
            position = True
        elif position and (es < em or r > rsi_max):
            trades.append({
                "date": data[i]["date"], "action": "SELL",
                "price": data[i]["close"], "rsi": round(r, 2),
            })
            position = False

    return trades


def n_day_high_breakout_strategy(
    data: List[Dict],
    n_days: int = 63,
    volume_ratio_min: float = 1.5,
) -> List[Dict]:
    """
    N-Day High Breakout  (William O'Neil / CANSLIM style)
    -----------------------------------------------------
    Setup:  Stock consolidates below an N-day high.
    Entry:  Close > N-day high AND today's volume ≥ volume_ratio_min × 20-day avg vol.
    Exit:   Close drops below N-day low (mean-reversion stop).

    Win rate: typically 55-65 % when combined with a bull-market regime filter.
    """
    if len(data) < n_days + 20:
        return []

    closes = [d["close"] for d in data]
    volumes = [d["volume"] for d in data]
    vol_ratio = compute_volume_ratio(volumes, period=20)

    trades: List[Dict] = []
    position = False

    for i in range(n_days, len(data)):
        window = data[i - n_days : i]
        n_day_high = max(d["high"] for d in window)
        n_day_low = min(d["low"] for d in window)
        vr = vol_ratio[i]

        if not position:
            if closes[i] > n_day_high and vr is not None and vr >= volume_ratio_min:
                trades.append({
                    "date": data[i]["date"], "action": "BUY",
                    "price": closes[i],
                    "n_day_high": round(n_day_high, 2),
                    "volume_ratio": round(vr, 4),
                })
                position = True
        else:
            if closes[i] < n_day_low:
                trades.append({
                    "date": data[i]["date"], "action": "SELL",
                    "price": closes[i],
                    "n_day_low": round(n_day_low, 2),
                })
                position = False

    return trades


def rsi_bollinger_combo_strategy(
    data: List[Dict],
    rsi_buy: float = 32.0,
    rsi_sell: float = 68.0,
) -> List[Dict]:
    """
    RSI + Bollinger Band Mean Reversion  (Larry Connors RSI2 variant)
    -----------------------------------------------------------------
    Entry:  RSI < rsi_buy AND price touches/crosses lower Bollinger band.
    Exit:   RSI > rsi_sell OR price crosses above middle band.

    Win rate: typically 65-75 % — one of the highest win-rate mean-reversion setups.
    """
    enriched = compute_all(data)
    trades: List[Dict] = []
    position = False

    for bar in enriched:
        rsi = bar.get("rsi_14")
        pct_b = bar.get("bb_pct_b")
        if rsi is None or pct_b is None:
            continue

        if not position and rsi < rsi_buy and pct_b <= 0.10:
            trades.append({
                "date": bar["date"], "action": "BUY",
                "price": bar["close"],
                "rsi": round(rsi, 2), "bb_pct_b": round(pct_b, 4),
            })
            position = True
        elif position and (rsi > rsi_sell or pct_b >= 0.50):
            trades.append({
                "date": bar["date"], "action": "SELL",
                "price": bar["close"],
                "rsi": round(rsi, 2), "bb_pct_b": round(pct_b, 4),
            })
            position = False

    return trades


# =========================================================================
# Strategy registry  (name → (fn, parameter_grid))
# =========================================================================

def _strategy_registry() -> Dict[str, Dict]:
    """
    Returns a dict of:
      strategy_name → {
          "fn": callable(data, **params) → List[Dict],
          "grid": List[Dict[str, Any]],    # each dict is one param combo
          "description": str,
      }
    """
    # Helper: cartesian product without itertools
    def _product(grid_dict):
        keys = list(grid_dict.keys())
        values = list(grid_dict.values())
        combos = [{}]
        for k, vals in zip(keys, values):
            combos = [{**c, k: v} for c in combos for v in vals]
        return combos

    return {
        "sma_crossover": {
            "fn": lambda data, short_period=10, long_period=30, **_:
                BacktestEngine(data)._BacktestEngine__sma_wrap(data, short_period, long_period),
            "grid": _product({"short_period": [5, 10, 15, 20], "long_period": [20, 30, 50, 100]}),
            "description": "SMA Crossover (classic, low win rate but strong trend capture)",
        },
        "rsi_bb_mean_reversion": {
            "fn": lambda data, rsi_buy=32.0, rsi_sell=68.0, **_:
                rsi_bollinger_combo_strategy(data, rsi_buy, rsi_sell),
            "grid": _product({"rsi_buy": [25.0, 28.0, 30.0, 32.0, 35.0], "rsi_sell": [65.0, 68.0, 70.0, 72.0]}),
            "description": "RSI + Bollinger Mean Reversion — highest win rate (65-75%)",
        },
        "macd_momentum": {
            "fn": lambda data, volume_ratio_threshold=1.2, **_:
                BacktestEngine(data).macd_momentum_strategy(data, volume_ratio_threshold=volume_ratio_threshold),
            "grid": _product({"volume_ratio_threshold": [1.0, 1.2, 1.5, 2.0]}),
            "description": "MACD Momentum with volume confirmation",
        },
        "bb_squeeze_breakout": {
            "fn": lambda data, squeeze_pct=20, **_:
                bb_squeeze_breakout_strategy(data, squeeze_pct=squeeze_pct),
            "grid": _product({"squeeze_pct": [10, 15, 20, 25]}),
            "description": "BB Squeeze Breakout — catches explosive moves after consolidation",
        },
        "ema_stack": {
            "fn": lambda data, rsi_max=72.0, **_:
                ema_stack_strategy(data, rsi_max=rsi_max),
            "grid": _product({"rsi_max": [68.0, 70.0, 72.0, 75.0]}),
            "description": "EMA Stack (8/21/50 aligned) — follows institutional trend structure",
        },
        "n_day_high_breakout": {
            "fn": lambda data, n_days=63, volume_ratio_min=1.5, **_:
                n_day_high_breakout_strategy(data, n_days=n_days, volume_ratio_min=volume_ratio_min),
            "grid": _product({"n_days": [21, 42, 63, 126], "volume_ratio_min": [1.2, 1.5, 2.0]}),
            "description": "N-Day High Breakout (O'Neil/CANSLIM) — momentum with volume confirmation",
        },
        "combined_signal": {
            "fn": lambda data, buy_threshold=0.5, **_:
                BacktestEngine(data).combined_signal_strategy(data, buy_threshold=buy_threshold),
            "grid": _product({"buy_threshold": [0.35, 0.45, 0.50, 0.55, 0.65]}),
            "description": "Multi-indicator composite score (RSI + MACD + BB + Volume)",
        },
    }


# Patch BacktestEngine with a thin wrapper so the registry lambda works
def _patch_backend():
    """Add __sma_wrap helper to BacktestEngine for registry use."""
    def _sma_wrap(self, data, short_period, long_period):
        return self.moving_average_crossover_strategy(data, short_period, long_period)
    BacktestEngine._BacktestEngine__sma_wrap = _sma_wrap


_patch_backend()


# =========================================================================
# Performance helpers
# =========================================================================

def _trades_in_window(trades: List[Dict], start_date: str, end_date: str) -> List[Dict]:
    """Filter trades whose date falls within [start_date, end_date]."""
    return [t for t in trades if start_date <= t["date"] <= end_date]


def _build_equity_curve(
    trades: List[Dict],
    initial_capital: float,
    commission: float,
    slippage: float,
) -> Tuple[List[str], List[float]]:
    """
    Build a list of (date, equity) points from a trade list.
    Returns (dates, equity_values) — one point per round-trip.
    """
    dates = ["start"]
    equity = [initial_capital]
    capital = initial_capital
    shares = 0.0
    cost_basis = 0.0

    for t in trades:
        price = t["price"]
        action = t["action"]

        if action == "BUY":
            eff = price * (1 + slippage)
            cost_basis = capital - commission
            shares = cost_basis / eff
            capital = 0.0
        elif action == "SELL" and shares > 0:
            eff = price * (1 - slippage)
            capital = shares * eff - commission
            shares = 0.0
            cost_basis = 0.0
            dates.append(t["date"])
            equity.append(round(capital, 2))

    # Mark to market if still holding at end
    if shares > 0 and trades:
        capital = shares * trades[-1]["price"]
        dates.append(trades[-1]["date"])
        equity.append(round(capital, 2))

    return dates, equity


def _build_daily_equity_curve(
    oos_trades: List[Dict],
    oos_data: List[Dict],
    initial_capital: float,
    commission: float,
    slippage: float,
) -> List[float]:
    """
    Build a daily equity value for every bar in oos_data.
    Cash positions stay flat; open positions mark-to-market each bar.
    This gives a smooth, date-aligned curve for the graph.
    """
    # Index trades by date (first 10 chars = YYYY-MM-DD)
    trade_by_date: Dict[str, Dict] = {}
    for t in oos_trades:
        trade_by_date[t["date"][:10]] = t

    equity: List[float] = []
    capital = initial_capital
    shares = 0.0

    for bar in oos_data:
        date_key = bar["date"][:10]
        trade = trade_by_date.get(date_key)

        if trade:
            action = trade["action"]
            price = trade["price"]
            if action == "BUY" and shares == 0:
                eff = price * (1 + slippage)
                shares = (capital - commission) / eff
                capital = 0.0
            elif action == "SELL" and shares > 0:
                eff = price * (1 - slippage)
                capital = shares * eff - commission
                shares = 0.0

        # Mark-to-market
        if shares > 0:
            equity.append(round(shares * bar["close"], 2))
        else:
            equity.append(round(capital, 2))

    return equity


# =========================================================================
# Core OOS backtest engine
# =========================================================================

class OOSBacktest:
    """
    Walk-Forward Holdout Backtest.

    Parameters
    ----------
    data:            Full OHLCV bar list (recommended: 8-10 years of daily bars).
    is_pct:          Fraction used as In-Sample training window (default 0.6 = 6 of 10 years).
    initial_capital: Starting capital for each strategy simulation.
    commission:      Flat commission per trade.
    slippage:        Slippage fraction per trade.
    """

    def __init__(
        self,
        data: List[Dict],
        is_pct: float = 0.6,
        initial_capital: float = 10_000.0,
        commission: float = 1.0,
        slippage: float = 0.0005,
    ) -> None:
        self.data = data
        self.is_pct = is_pct
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

        split_idx = int(len(data) * is_pct)
        self.is_data = data[:split_idx]
        self.oos_data = data[split_idx:]
        self.is_end_date = self.is_data[-1]["date"] if self.is_data else ""
        self.oos_start_date = self.oos_data[0]["date"] if self.oos_data else ""
        self.oos_end_date = self.oos_data[-1]["date"] if self.oos_data else ""

    # ------------------------------------------------------------------
    # Grid search (IS only)
    # ------------------------------------------------------------------

    def _grid_search(
        self,
        strategy_name: str,
        strategy_info: Dict,
    ) -> Tuple[Dict, float]:
        """
        Try every param combo from the grid, running on ALL data but scoring
        only on IS trades.  Returns (best_params, best_is_sharpe).
        """
        fn = strategy_info["fn"]
        grid = strategy_info["grid"]
        engine = BacktestEngine(self.data)  # full data for indicator warmup

        best_params: Dict = grid[0]
        best_sharpe = float("-inf")

        for params in grid:
            try:
                trades = fn(self.data, **params)
            except Exception:
                continue

            is_trades = _trades_in_window(trades, self.data[0]["date"], self.is_end_date)
            perf = engine.calculate_performance(
                is_trades, self.initial_capital, self.commission, self.slippage
            )
            sharpe = perf.get("sharpe_ratio", 0.0)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params

        return best_params, best_sharpe

    # ------------------------------------------------------------------
    # Run a strategy and compute IS + OOS metrics
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        strategy_name: str,
        strategy_info: Dict,
        best_params: Dict,
    ) -> Dict:
        """
        Run strategy on all data with best_params, split trades into IS/OOS,
        and compute performance for each window.
        """
        fn = strategy_info["fn"]
        engine = BacktestEngine(self.data)

        try:
            all_trades = fn(self.data, **best_params)
        except Exception as e:
            return {"error": str(e)}

        is_trades = _trades_in_window(all_trades, self.data[0]["date"], self.is_end_date)
        oos_trades = _trades_in_window(all_trades, self.oos_start_date, self.oos_end_date)

        is_perf = engine.calculate_performance(
            is_trades, self.initial_capital, self.commission, self.slippage
        )
        oos_perf = engine.calculate_performance(
            oos_trades, self.initial_capital, self.commission, self.slippage
        )

        # Build day-by-day equity curve aligned to OOS bars
        oos_equity = _build_daily_equity_curve(
            oos_trades, self.oos_data, self.initial_capital, self.commission, self.slippage
        )

        return {
            "best_params": best_params,
            "is_performance": is_perf,
            "oos_performance": oos_perf,
            "oos_equity_daily": oos_equity,      # len == len(oos_data), aligned by bar index
            "oos_trades": oos_trades,             # kept for graph annotation
            "all_trades_count": len(all_trades),
            "description": strategy_info["description"],
        }

    # ------------------------------------------------------------------
    # Buy-and-hold benchmark
    # ------------------------------------------------------------------

    def _buy_and_hold(self) -> Dict:
        if not self.oos_data:
            return {}

        oos_start_price = self.oos_data[0]["open"]
        oos_end_price = self.oos_data[-1]["close"]
        shares = (self.initial_capital - self.commission) / oos_start_price
        final = shares * oos_end_price - self.commission
        ret = (final - self.initial_capital) / self.initial_capital * 100

        # Day-by-day equity for smooth curve
        equity_dates = [d["date"] for d in self.oos_data]
        equity_values = [
            round(shares * d["close"], 2) for d in self.oos_data
        ]

        # Max drawdown
        peak = equity_values[0]
        max_dd = 0.0
        for v in equity_values:
            peak = max(peak, v)
            dd = (peak - v) / peak * 100 if peak else 0
            max_dd = max(max_dd, dd)

        # Annualised return
        try:
            s_dt = datetime.strptime(self.oos_data[0]["date"][:10], "%Y-%m-%d")
            e_dt = datetime.strptime(self.oos_data[-1]["date"][:10], "%Y-%m-%d")
            years = max((e_dt - s_dt).days / 365.25, 0.01)
            ann = ((1 + ret / 100) ** (1 / years) - 1) * 100
        except Exception:
            ann = 0.0

        return {
            "description": "Buy and Hold (benchmark)",
            "oos_performance": {
                "total_return": round(ret, 2),
                "final_capital": round(final, 2),
                "max_drawdown_pct": round(max_dd, 2),
                "sharpe_ratio": 0.0,
                "total_trades": 1,
                "win_rate": 100.0 if ret > 0 else 0.0,
            },
            "annualized_return": round(ann, 2),
            "oos_equity_daily": equity_values,    # daily-aligned, len == len(oos_data)
            "oos_equity_values": equity_values,   # alias kept for compatibility
            "oos_equity_dates": equity_dates,
        }

    # ------------------------------------------------------------------
    # SPY 10 % / year benchmark line
    # ------------------------------------------------------------------

    def _spy_benchmark_equity(self) -> Tuple[List[str], List[float]]:
        """Straight-line 10 %/year equity growth over the OOS period."""
        if not self.oos_data:
            return [], []
        try:
            start_dt = datetime.strptime(self.oos_data[0]["date"][:10], "%Y-%m-%d")
            dates, equity = [], []
            for d in self.oos_data:
                dt = datetime.strptime(d["date"][:10], "%Y-%m-%d")
                years = (dt - start_dt).days / 365.25
                equity.append(round(self.initial_capital * (1.10 ** years), 2))
                dates.append(d["date"])
            return dates, equity
        except Exception:
            return [], []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, strategies: Optional[List[str]] = None) -> Dict:
        """
        Run the full IS optimisation → OOS simulation pipeline.

        Parameters
        ----------
        strategies: Optional list of strategy names to run. Default = all.

        Returns
        -------
        Full results dict including per-strategy metrics, equity curves,
        formatted summary table, and data period info.
        """
        registry = _strategy_registry()
        if strategies:
            registry = {k: v for k, v in registry.items() if k in strategies}

        results: Dict[str, Any] = {}

        for name, info in registry.items():
            best_params, best_is_sharpe = self._grid_search(name, info)
            result = self._evaluate(name, info, best_params)
            result["best_is_sharpe_from_grid"] = round(best_is_sharpe, 4)
            results[name] = result

        results["buy_and_hold"] = self._buy_and_hold()
        spy_dates, spy_equity = self._spy_benchmark_equity()

        # Annualised OOS return for each strategy
        try:
            s_dt = datetime.strptime(self.oos_data[0]["date"][:10], "%Y-%m-%d")
            e_dt = datetime.strptime(self.oos_data[-1]["date"][:10], "%Y-%m-%d")
            oos_years = max((e_dt - s_dt).days / 365.25, 0.01)
        except Exception:
            oos_years = 1.0

        for name, res in results.items():
            if "oos_performance" in res:
                ret = res["oos_performance"].get("total_return", 0.0)
                res["oos_annualized_return"] = round(
                    ((1 + ret / 100) ** (1 / oos_years) - 1) * 100, 2
                )

        return {
            "terminology": (
                "Walk-Forward Holdout Backtest: "
                f"IS={round(self.is_pct*100)}% ({len(self.is_data)} bars, "
                f"up to {self.is_end_date[:10]}), "
                f"OOS={round((1-self.is_pct)*100)}% ({len(self.oos_data)} bars, "
                f"{self.oos_start_date[:10]} → {self.oos_end_date[:10]})"
            ),
            "data_summary": {
                "total_bars": len(self.data),
                "is_bars": len(self.is_data),
                "oos_bars": len(self.oos_data),
                "is_period": {"start": self.data[0]["date"][:10], "end": self.is_end_date[:10]},
                "oos_period": {"start": self.oos_start_date[:10], "end": self.oos_end_date[:10]},
                "oos_years": round(oos_years, 2),
                "initial_capital": self.initial_capital,
            },
            "strategies": results,
            "spy_benchmark": {
                "dates": spy_dates[::5],          # downsample for JSON size
                "equity": spy_equity[::5],
                "annual_pct": 10.0,
            },
            "summary_table": self.format_table(results, oos_years),
        }

    # ------------------------------------------------------------------
    # Text table
    # ------------------------------------------------------------------

    @staticmethod
    def format_table(results: Dict, oos_years: float = 1.0) -> str:
        header = (
            f"{'Strategy':<24} | {'IS Ret':>7} | {'IS Sharpe':>9} | "
            f"{'OOS Ret':>8} | {'OOS Ann.':>8} | {'OOS Sharpe':>10} | "
            f"{'Max DD':>7} | {'Trades':>7} | {'Win%':>6}"
        )
        sep = "-" * len(header)
        rows = [sep, header, sep]

        order = sorted(
            [k for k in results if k != "buy_and_hold"],
            key=lambda k: results[k].get("oos_performance", {}).get("sharpe_ratio", -999),
            reverse=True,
        )
        order.append("buy_and_hold")

        for name in order:
            res = results.get(name, {})
            is_p = res.get("is_performance", {})
            oos_p = res.get("oos_performance", {})
            ann = res.get("oos_annualized_return", 0.0)

            is_ret  = f"{is_p.get('total_return', 0):+.1f}%" if is_p else "  n/a"
            is_sh   = f"{is_p.get('sharpe_ratio', 0):.2f}"  if is_p else "  n/a"
            oos_ret = f"{oos_p.get('total_return', 0):+.1f}%"
            oos_ann = f"{ann:+.1f}%"
            oos_sh  = f"{oos_p.get('sharpe_ratio', 0):.2f}"
            dd      = f"{oos_p.get('max_drawdown_pct', 0):.1f}%"
            trades  = str(oos_p.get("total_trades", 0))
            win     = f"{oos_p.get('win_rate', 0):.0f}%"

            row = (
                f"{name:<24} | {is_ret:>7} | {is_sh:>9} | "
                f"{oos_ret:>8} | {oos_ann:>8} | {oos_sh:>10} | "
                f"{dd:>7} | {trades:>7} | {win:>6}"
            )
            rows.append(row)

        rows.append(sep)
        rows.append(
            "IS = In-Sample (training)  |  OOS = Out-of-Sample (blind simulation)  "
            "|  Ann. = annualised OOS return"
        )
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def generate_graph(
        self,
        results: Dict,
        filepath: str = "graphs/oos_backtest.png",
        symbol: str = "",
    ) -> str:
        """
        Two-panel figure with a shared daily bar-index X axis:
          Top:    OOS equity curves — all strategies + buy-and-hold + SPY 10%/yr line
          Bottom: OOS drawdown curves
        X axis = bar index (0 … len(oos_data)-1) so all curves align by date.
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        n_bars = len(self.oos_data)
        # X-tick labels: one every ~20 bars
        tick_step = max(1, n_bars // 8)
        tick_positions = list(range(0, n_bars, tick_step))
        tick_labels = [self.oos_data[i]["date"][:10] for i in tick_positions]

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.12)
        ax_eq = fig.add_subplot(gs[0])
        ax_dd = fig.add_subplot(gs[1], sharex=ax_eq)

        palette = [
            "#2196F3", "#4CAF50", "#FF5722", "#9C27B0",
            "#FF9800", "#00BCD4", "#E91E63", "#795548",
        ]

        def _drawdown(eq):
            dd, peak = [], eq[0] if eq else 1
            for v in eq:
                peak = max(peak, v)
                dd.append(-(peak - v) / peak * 100 if peak else 0)
            return dd

        x = list(range(n_bars))
        col_idx = 0
        strategy_results = results.get("strategies", results)

        for name, res in strategy_results.items():
            if name == "buy_and_hold":
                continue
            eq = res.get("oos_equity_daily", [])
            if len(eq) != n_bars:
                continue
            color = palette[col_idx % len(palette)]
            col_idx += 1
            oos_ret = res.get("oos_performance", {}).get("total_return", 0)
            ax_eq.plot(x, eq, color=color, linewidth=1.5,
                       label=f"{name}  ({oos_ret:+.1f}%)")
            ax_dd.plot(x, _drawdown(eq), color=color, linewidth=1.0)

        # Buy-and-hold
        bah = strategy_results.get("buy_and_hold", {})
        bah_eq = bah.get("oos_equity_values", [])
        if bah_eq and len(bah_eq) == n_bars:
            bah_ret = bah.get("oos_performance", {}).get("total_return", 0)
            ax_eq.plot(x, bah_eq, color="black", linewidth=2, linestyle="--",
                       label=f"Buy & Hold  ({bah_ret:+.1f}%)", alpha=0.7)
            ax_dd.plot(x, _drawdown(bah_eq), color="black", linewidth=1,
                       linestyle="--", alpha=0.6)

        # SPY 10%/yr straight line
        _, spy_eq = self._spy_benchmark_equity()
        if spy_eq and len(spy_eq) == n_bars:
            ax_eq.plot(x, spy_eq, color="gray", linewidth=1.5, linestyle=":",
                       label="SPY 10%/yr benchmark", alpha=0.8)

        ax_eq.axhline(self.initial_capital, color="gray", linewidth=0.8,
                      linestyle="--", alpha=0.4)

        title_sym = f" — {symbol.upper()}" if symbol else ""
        ax_eq.set_title(
            f"Walk-Forward OOS Backtest{title_sym}\n"
            f"OOS: {self.oos_start_date[:10]} → {self.oos_end_date[:10]}  "
            f"({n_bars} bars)  |  IS used for parameter optimisation only",
            fontsize=11,
        )
        ax_eq.set_ylabel("Portfolio Value ($)")
        ax_eq.legend(loc="upper left", fontsize=8, ncol=2)
        ax_eq.grid(True, alpha=0.3)
        plt.setp(ax_eq.get_xticklabels(), visible=False)

        ax_dd.set_ylabel("Drawdown (%)")
        ax_dd.set_xlabel("Date")
        ax_dd.set_xticks(tick_positions)
        ax_dd.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)
        ax_dd.axhline(0, color="gray", linewidth=0.5)
        ax_dd.grid(True, alpha=0.3)

        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        return filepath
