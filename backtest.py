from fastapi import APIRouter, Path, Query
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
import json
import os
import math
from typing import Dict, List, Optional, Tuple
from datetime import datetime

router = APIRouter()


class BacktestEngine:
    """
    Engine for backtesting trading strategies on historical data.
    Supports:
      - Train/validation split
      - Walk-forward analysis with sliding windows
      - Per-trade P&L tracking
      - Sharpe ratio, max drawdown, Sortino ratio
      - Transaction cost modeling (commission + slippage)
    """

    def __init__(self, data: List[Dict], train_split: float = 0.8):
        self.full_data = data
        self.train_split = train_split

        split_index = int(len(data) * train_split)
        self.train_data = data[:split_index]
        self.validation_data = data[split_index:]

    def get_data_summary(self) -> Dict:
        return {
            "total_data_points": len(self.full_data),
            "train_data_points": len(self.train_data),
            "validation_data_points": len(self.validation_data),
            "train_period": {
                "start": self.train_data[0]["date"] if self.train_data else None,
                "end": self.train_data[-1]["date"] if self.train_data else None,
            },
            "validation_period": {
                "start": self.validation_data[0]["date"] if self.validation_data else None,
                "end": self.validation_data[-1]["date"] if self.validation_data else None,
            },
            "train_split_percentage": self.train_split * 100,
        }

    # ------------------------------------------------------------------
    # Strategy
    # ------------------------------------------------------------------
    def moving_average_crossover_strategy(
        self,
        data: List[Dict],
        short_period: int = 10,
        long_period: int = 30,
        allow_short: bool = False,
    ) -> List[Dict]:
        """
        SMA Crossover: BUY when short MA crosses above long MA,
        SELL when it crosses below.

        If allow_short is True, also opens SHORT positions when the
        short MA crosses below the long MA (profits in bear markets).
        """
        if len(data) < long_period + 1:
            return []

        trades: List[Dict] = []
        position = None  # None, "LONG", or "SHORT"

        for i in range(long_period, len(data)):
            short_ma = sum(d["close"] for d in data[i - short_period : i]) / short_period
            long_ma = sum(d["close"] for d in data[i - long_period : i]) / long_period

            prev_short_ma = (
                sum(d["close"] for d in data[i - short_period - 1 : i - 1]) / short_period
            )
            prev_long_ma = (
                sum(d["close"] for d in data[i - long_period - 1 : i - 1]) / long_period
            )

            # --- Bullish crossover ---
            if prev_short_ma <= prev_long_ma and short_ma > long_ma:
                # Close short first if we have one
                if position == "SHORT" and allow_short:
                    trades.append(
                        {
                            "date": data[i]["date"],
                            "action": "COVER",
                            "price": data[i]["close"],
                            "short_ma": round(short_ma, 4),
                            "long_ma": round(long_ma, 4),
                        }
                    )
                if position != "LONG":
                    trades.append(
                        {
                            "date": data[i]["date"],
                            "action": "BUY",
                            "price": data[i]["close"],
                            "short_ma": round(short_ma, 4),
                            "long_ma": round(long_ma, 4),
                        }
                    )
                    position = "LONG"

            # --- Bearish crossover ---
            elif prev_short_ma >= prev_long_ma and short_ma < long_ma:
                # Close long first if we have one
                if position == "LONG":
                    trades.append(
                        {
                            "date": data[i]["date"],
                            "action": "SELL",
                            "price": data[i]["close"],
                            "short_ma": round(short_ma, 4),
                            "long_ma": round(long_ma, 4),
                        }
                    )
                if allow_short and position != "SHORT":
                    trades.append(
                        {
                            "date": data[i]["date"],
                            "action": "SHORT",
                            "price": data[i]["close"],
                            "short_ma": round(short_ma, 4),
                            "long_ma": round(long_ma, 4),
                        }
                    )
                    position = "SHORT"
                else:
                    position = None

        return trades
    
    # ------------------------------------------------------------------
    # Performance (fixed per-trade P&L, added Sharpe / drawdown / costs)
    # ------------------------------------------------------------------
    def calculate_performance(
        self,
        trades: List[Dict],
        initial_capital: float = 10000.0,
        commission_per_trade: float = 1.0,
        slippage_pct: float = 0.0005,
    ) -> Dict:
        """
        Calculate performance metrics from a list of trades.

        Improvements over the original:
          1. Per-trade P&L is measured against the actual buy price (not initial_capital).
          2. Models commission (flat $ per trade) and slippage (% of trade value).
          3. Computes Sharpe ratio, Sortino ratio, and max drawdown.
        """
        empty_result = {
            "total_trades": 0,
            "completed_trades": 0,
            "profitable_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "total_return_dollars": 0.0,
            "final_capital": initial_capital,
            "initial_capital": initial_capital,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "total_commissions": 0.0,
            "total_slippage_cost": 0.0,
            "per_trade_returns": [],
        }

        if not trades or len(trades) < 2:
            return empty_result

        capital = initial_capital
        shares = 0.0
        buy_cost_basis = 0.0  # tracks the actual capital spent on the current position
        short_entry_price = 0.0  # tracks entry price for short positions
        profitable_trades = 0
        losing_trades = 0
        per_trade_returns: List[float] = []  # percentage return per round-trip
        total_commissions = 0.0
        total_slippage = 0.0

        # For max-drawdown tracking we record the equity curve at each trade event
        equity_curve: List[float] = [initial_capital]

        for trade in trades:
            price = trade["price"]

            if trade["action"] == "BUY":
                # Apply slippage: price moves against us on a buy
                effective_price = price * (1 + slippage_pct)
                commission = commission_per_trade
                total_commissions += commission
                total_slippage += (effective_price - price) * (capital / effective_price) if effective_price else 0

                buy_cost_basis = capital - commission  # net cash available after commission
                shares = buy_cost_basis / effective_price
                capital = 0.0
                equity_curve.append(shares * price)  # mark-to-market at raw price

            elif trade["action"] == "SELL" and shares > 0:
                # Apply slippage: price moves against us on a sell
                effective_price = price * (1 - slippage_pct)
                commission = commission_per_trade
                total_commissions += commission
                total_slippage += (price - effective_price) * shares if price else 0

                sell_proceeds = shares * effective_price - commission
                trade_return_pct = ((sell_proceeds - buy_cost_basis) / buy_cost_basis) * 100 if buy_cost_basis else 0.0

                per_trade_returns.append(trade_return_pct)

                if sell_proceeds > buy_cost_basis:
                    profitable_trades += 1
                else:
                    losing_trades += 1

                capital = sell_proceeds
                shares = 0.0
                buy_cost_basis = 0.0
                equity_curve.append(capital)

            elif trade["action"] == "SHORT":
                # Open short: we "sell" shares we don't own at current price
                effective_price = price * (1 - slippage_pct)
                commission = commission_per_trade
                total_commissions += commission
                total_slippage += (price - effective_price) * (capital / price) if price else 0

                buy_cost_basis = capital - commission  # capital committed
                shares = buy_cost_basis / price  # notional shares shorted
                short_entry_price = effective_price
                capital = 0.0
                equity_curve.append(buy_cost_basis)

            elif trade["action"] == "COVER" and shares > 0:
                # Close short: profit = (entry_price - exit_price) * shares
                effective_price = price * (1 + slippage_pct)
                commission = commission_per_trade
                total_commissions += commission
                total_slippage += (effective_price - price) * shares if price else 0

                cover_cost = shares * effective_price + commission
                short_proceeds = (shares * short_entry_price) - cover_cost + buy_cost_basis
                trade_return_pct = ((short_proceeds - buy_cost_basis) / buy_cost_basis) * 100 if buy_cost_basis else 0.0

                per_trade_returns.append(trade_return_pct)

                if short_proceeds > buy_cost_basis:
                    profitable_trades += 1
                else:
                    losing_trades += 1

                capital = short_proceeds
                shares = 0.0
                buy_cost_basis = 0.0
                equity_curve.append(capital)

        # If still holding, mark to market (no slippage—position not yet closed)
        if shares > 0 and trades:
            capital = shares * trades[-1]["price"]
            equity_curve.append(capital)

        # ---------- Derived metrics ----------
        total_completed = profitable_trades + losing_trades
        total_return = ((capital - initial_capital) / initial_capital) * 100

        # Max drawdown from equity curve
        max_drawdown_pct = self._max_drawdown(equity_curve)

        # Sharpe & Sortino (annualised, assuming ~252 trading days and trades are roughly evenly spaced)
        sharpe = self._sharpe_ratio(per_trade_returns)
        sortino = self._sortino_ratio(per_trade_returns)

        return {
            "total_trades": len([t for t in trades if t["action"] == "BUY"]),
            "completed_trades": total_completed,
            "profitable_trades": profitable_trades,
            "losing_trades": losing_trades,
            "win_rate": round((profitable_trades / total_completed * 100) if total_completed > 0 else 0.0, 2),
            "initial_capital": initial_capital,
            "final_capital": round(capital, 2),
            "total_return": round(total_return, 2),
            "total_return_dollars": round(capital - initial_capital, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "total_commissions": round(total_commissions, 2),
            "total_slippage_cost": round(total_slippage, 2),
            "per_trade_returns": [round(r, 2) for r in per_trade_returns],
        }

    # ------------------------------------------------------------------
    # Risk metrics helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _max_drawdown(equity_curve: List[float]) -> float:
        """Return maximum peak-to-trough drawdown as a positive percentage."""
        if len(equity_curve) < 2:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for value in equity_curve[1:]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100 if peak else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @staticmethod
    def _sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Annualised Sharpe ratio from per-trade return percentages."""
        if len(returns) < 2:
            return 0.0
        mean_r = sum(returns) / len(returns) - risk_free_rate
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1))
        if std_r == 0:
            return 0.0
        # Annualise assuming ~20 round-trip trades/year as a rough proxy
        return (mean_r / std_r) * math.sqrt(len(returns))

    @staticmethod
    def _sortino_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Sortino ratio: like Sharpe but only penalises downside volatility."""
        if len(returns) < 2:
            return 0.0
        mean_r = sum(returns) / len(returns) - risk_free_rate
        downside = [r for r in returns if r < 0]
        if not downside:
            return float("inf") if mean_r > 0 else 0.0
        downside_std = math.sqrt(sum(r ** 2 for r in downside) / len(downside))
        if downside_std == 0:
            return 0.0
        return (mean_r / downside_std) * math.sqrt(len(returns))

    # ------------------------------------------------------------------
    # Walk-forward analysis
    # ------------------------------------------------------------------
    def walk_forward_analysis(
        self,
        n_windows: int = 5,
        short_period: int = 10,
        long_period: int = 30,
        initial_capital: float = 10000.0,
        commission_per_trade: float = 1.0,
        slippage_pct: float = 0.0005,
    ) -> Dict:
        """
        Walk-forward analysis: slide a train/test window across the full dataset.

        The dataset is divided into `n_windows` equal folds.  For each fold *i*
        (starting from 1), all data before fold *i* is used for training (parameter
        fitting could go here) and fold *i* is used for out-of-sample testing.

        Returns per-window results and aggregate statistics.
        """
        data = self.full_data
        fold_size = len(data) // n_windows

        if fold_size < long_period + 1:
            return {"error": f"Not enough data for {n_windows} walk-forward windows with long_period={long_period}"}

        windows: List[Dict] = []
        all_oos_returns: List[float] = []

        for i in range(1, n_windows):
            train_end = fold_size * i
            test_end = min(fold_size * (i + 1), len(data))

            train_slice = data[:train_end]
            test_slice = data[train_end:test_end]

            if len(test_slice) < long_period + 1:
                continue

            # --- optional: optimise parameters on train_slice ---
            # For now we use the supplied parameters; a future enhancement
            # could grid-search here per window.

            train_trades = self.moving_average_crossover_strategy(train_slice, short_period, long_period)
            test_trades = self.moving_average_crossover_strategy(test_slice, short_period, long_period)

            train_perf = self.calculate_performance(train_trades, initial_capital, commission_per_trade, slippage_pct)
            test_perf = self.calculate_performance(test_trades, initial_capital, commission_per_trade, slippage_pct)

            all_oos_returns.append(test_perf["total_return"])

            windows.append(
                {
                    "window": i,
                    "train_points": len(train_slice),
                    "test_points": len(test_slice),
                    "train_period": {
                        "start": train_slice[0]["date"],
                        "end": train_slice[-1]["date"],
                    },
                    "test_period": {
                        "start": test_slice[0]["date"],
                        "end": test_slice[-1]["date"],
                    },
                    "train_performance": train_perf,
                    "test_performance": test_perf,
                }
            )

        avg_oos_return = sum(all_oos_returns) / len(all_oos_returns) if all_oos_returns else 0.0
        std_oos_return = (
            math.sqrt(sum((r - avg_oos_return) ** 2 for r in all_oos_returns) / len(all_oos_returns))
            if all_oos_returns
            else 0.0
        )

        return {
            "n_windows": len(windows),
            "parameters": {
                "short_period": short_period,
                "long_period": long_period,
                "initial_capital": initial_capital,
                "commission_per_trade": commission_per_trade,
                "slippage_pct": slippage_pct,
            },
            "aggregate": {
                "avg_oos_return": round(avg_oos_return, 2),
                "std_oos_return": round(std_oos_return, 2),
                "min_oos_return": round(min(all_oos_returns), 2) if all_oos_returns else 0.0,
                "max_oos_return": round(max(all_oos_returns), 2) if all_oos_returns else 0.0,
                "all_oos_returns": [round(r, 2) for r in all_oos_returns],
            },
            "windows": windows,
        }

    # ------------------------------------------------------------------
    # Standard single-split backtest (now with costs)
    # ------------------------------------------------------------------
    def run_backtest(
        self,
        strategy_name: str = "moving_average_crossover",
        short_period: int = 10,
        long_period: int = 30,
        initial_capital: float = 10000.0,
        commission_per_trade: float = 1.0,
        slippage_pct: float = 0.0005,
        allow_short: bool = False,
    ) -> Dict:
        if strategy_name == "moving_average_crossover":
            train_trades = self.moving_average_crossover_strategy(self.train_data, short_period, long_period, allow_short)
            validation_trades = self.moving_average_crossover_strategy(self.validation_data, short_period, long_period, allow_short)
        else:
            return {"error": f"Unknown strategy: {strategy_name}"}

        train_performance = self.calculate_performance(train_trades, initial_capital, commission_per_trade, slippage_pct)
        validation_performance = self.calculate_performance(validation_trades, initial_capital, commission_per_trade, slippage_pct)

        return {
            "strategy": strategy_name,
            "parameters": {
                "short_period": short_period,
                "long_period": long_period,
                "initial_capital": initial_capital,
                "commission_per_trade": commission_per_trade,
                "slippage_pct": slippage_pct,
                "allow_short": allow_short,
            },
            "data_summary": self.get_data_summary(),
            "training_results": {
                "trades": train_trades,
                "performance": train_performance,
            },
            "validation_results": {
                "trades": validation_trades,
                "performance": validation_performance,
            },
            "summary": {
                "train_return": train_performance["total_return"],
                "validation_return": validation_performance["total_return"],
                "performance_difference": round(
                    train_performance["total_return"] - validation_performance["total_return"], 2
                ),
                "train_sharpe": train_performance["sharpe_ratio"],
                "validation_sharpe": validation_performance["sharpe_ratio"],
                "train_max_drawdown": train_performance["max_drawdown_pct"],
                "validation_max_drawdown": validation_performance["max_drawdown_pct"],
            },
        }


# ======================================================================
# API Endpoints
# ======================================================================


@router.post("/backtest/{file_id}")
async def run_backtest(
    file_id: str = Path(..., description="File ID of saved historical data"),
    strategy: str = Query("moving_average_crossover", description="Strategy name"),
    short_period: int = Query(10, description="Short MA period"),
    long_period: int = Query(30, description="Long MA period"),
    train_split: float = Query(0.8, description="Training data percentage (0.8 = 80%)"),
    initial_capital: float = Query(10000.0, description="Initial capital for backtest"),
    commission: float = Query(1.0, description="Commission per trade in dollars"),
    slippage: float = Query(0.0005, description="Slippage as a fraction (0.0005 = 0.05%)"),
    allow_short: bool = Query(False, description="Allow short selling on bearish crossovers"),
    output_dir: str = Query("historical_data", description="Directory with JSON files"),
):
    """
    Run a backtest on historical data from a saved JSON file.
    Splits data into training and validation periods.
    Models transaction costs (commission + slippage).
    """

    def execute_backtest():
        if not os.path.exists(output_dir):
            return {"error": "Historical data directory not found"}

        matching_files = [f for f in os.listdir(output_dir) if f.endswith(f"_{file_id}.json")]
        if not matching_files:
            return {"error": f"No file found with ID {file_id}"}

        filepath = os.path.join(output_dir, matching_files[0])
        with open(filepath, "r") as f:
            file_data = json.load(f)

        data = file_data.get("data", [])
        if not data:
            return {"error": "No data found in file"}

        engine = BacktestEngine(data, train_split=train_split)
        results = engine.run_backtest(
            strategy_name=strategy,
            short_period=short_period,
            long_period=long_period,
            initial_capital=initial_capital,
            commission_per_trade=commission,
            slippage_pct=slippage,
            allow_short=allow_short,
        )

        results["file_info"] = {
            "file_id": file_id,
            "symbol": file_data.get("symbol"),
            "duration": file_data.get("duration"),
            "bar_size": file_data.get("bar_size"),
        }
        return results

    result = await run_in_threadpool(execute_backtest)
    return JSONResponse(result)


@router.post("/walk_forward/{file_id}")
async def walk_forward(
    file_id: str = Path(..., description="File ID of saved historical data"),
    strategy: str = Query("moving_average_crossover", description="Strategy name"),
    short_period: int = Query(10, description="Short MA period"),
    long_period: int = Query(30, description="Long MA period"),
    n_windows: int = Query(5, description="Number of walk-forward windows"),
    initial_capital: float = Query(10000.0, description="Initial capital"),
    commission: float = Query(1.0, description="Commission per trade in dollars"),
    slippage: float = Query(0.0005, description="Slippage as a fraction"),
    output_dir: str = Query("historical_data", description="Directory with JSON files"),
):
    """
    Run walk-forward analysis: slides a train/test window across the dataset.
    Much more robust than a single train/validation split.
    """

    def execute_walk_forward():
        if not os.path.exists(output_dir):
            return {"error": "Historical data directory not found"}

        matching_files = [f for f in os.listdir(output_dir) if f.endswith(f"_{file_id}.json")]
        if not matching_files:
            return {"error": f"No file found with ID {file_id}"}

        filepath = os.path.join(output_dir, matching_files[0])
        with open(filepath, "r") as f:
            file_data = json.load(f)

        data = file_data.get("data", [])
        if not data:
            return {"error": "No data found in file"}

        engine = BacktestEngine(data)
        results = engine.walk_forward_analysis(
            n_windows=n_windows,
            short_period=short_period,
            long_period=long_period,
            initial_capital=initial_capital,
            commission_per_trade=commission,
            slippage_pct=slippage,
        )

        results["file_info"] = {
            "file_id": file_id,
            "symbol": file_data.get("symbol"),
            "duration": file_data.get("duration"),
            "bar_size": file_data.get("bar_size"),
        }
        return results

    result = await run_in_threadpool(execute_walk_forward)
    return JSONResponse(result)


@router.get("/optimize_strategy/{file_id}")
async def optimize_strategy(
    file_id: str = Path(..., description="File ID of saved historical data"),
    strategy: str = Query("moving_average_crossover", description="Strategy name"),
    train_split: float = Query(0.8, description="Training data percentage"),
    initial_capital: float = Query(10000.0, description="Initial capital"),
    commission: float = Query(1.0, description="Commission per trade"),
    slippage: float = Query(0.0005, description="Slippage fraction"),
    output_dir: str = Query("historical_data", description="Directory with JSON files"),
):
    """
    Optimize strategy parameters by grid-searching on training data only.
    Now ranks by Sharpe ratio instead of raw return to prefer risk-adjusted performance.
    """

    def find_optimal_params():
        if not os.path.exists(output_dir):
            return {"error": "Historical data directory not found"}

        matching_files = [f for f in os.listdir(output_dir) if f.endswith(f"_{file_id}.json")]
        if not matching_files:
            return {"error": f"No file found with ID {file_id}"}

        filepath = os.path.join(output_dir, matching_files[0])
        with open(filepath, "r") as f:
            file_data = json.load(f)

        data = file_data.get("data", [])
        if not data:
            return {"error": "No data found in file"}

        engine = BacktestEngine(data, train_split=train_split)

        best_params = None
        best_sharpe = float("-inf")
        results = []

        for short in [5, 10, 15, 20]:
            for long in [20, 30, 50, 100]:
                if short >= long:
                    continue

                trades = engine.moving_average_crossover_strategy(engine.train_data, short, long)
                performance = engine.calculate_performance(trades, initial_capital, commission, slippage)

                result = {
                    "short_period": short,
                    "long_period": long,
                    "return": performance["total_return"],
                    "sharpe_ratio": performance["sharpe_ratio"],
                    "sortino_ratio": performance["sortino_ratio"],
                    "max_drawdown_pct": performance["max_drawdown_pct"],
                    "win_rate": performance["win_rate"],
                    "total_trades": performance["total_trades"],
                    "total_commissions": performance["total_commissions"],
                }
                results.append(result)

                if performance["sharpe_ratio"] > best_sharpe:
                    best_sharpe = performance["sharpe_ratio"]
                    best_params = {
                        "short_period": short,
                        "long_period": long,
                        "performance": performance,
                    }

        # Also run the best params on validation data to check for overfitting
        validation_check = None
        if best_params:
            val_trades = engine.moving_average_crossover_strategy(
                engine.validation_data, best_params["short_period"], best_params["long_period"]
            )
            val_perf = engine.calculate_performance(val_trades, initial_capital, commission, slippage)
            validation_check = {
                "validation_return": val_perf["total_return"],
                "validation_sharpe": val_perf["sharpe_ratio"],
                "validation_max_drawdown": val_perf["max_drawdown_pct"],
                "overfit_warning": val_perf["total_return"] < best_params["performance"]["total_return"] * 0.5,
            }

        return {
            "file_id": file_id,
            "symbol": file_data.get("symbol"),
            "best_parameters": best_params,
            "validation_check": validation_check,
            "all_results": sorted(results, key=lambda x: x["sharpe_ratio"], reverse=True),
            "data_summary": engine.get_data_summary(),
        }

    result = await run_in_threadpool(find_optimal_params)
    return JSONResponse(result)