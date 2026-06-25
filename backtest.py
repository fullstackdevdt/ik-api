from fastapi import APIRouter, Path, Query
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
import json
import os
import math
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from indicators import compute_all
from regime import get_regime

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

    # ------------------------------------------------------------------
    # Additional strategies
    # ------------------------------------------------------------------

    def rsi_mean_reversion_strategy(
        self,
        data: List[Dict],
        rsi_buy: float = 30.0,
        rsi_sell: float = 70.0,
        require_bb_touch: bool = True,
    ) -> List[Dict]:
        """
        RSI Mean Reversion:
          BUY  when RSI < rsi_buy (oversold) AND optionally price ≤ lower Bollinger band.
          SELL when RSI > rsi_sell (overbought) OR price crosses above BB middle band.
        """
        enriched = compute_all(data)
        trades: List[Dict] = []
        position = False

        for i, bar in enumerate(enriched):
            rsi = bar.get("rsi_14")
            pct_b = bar.get("bb_pct_b")
            if rsi is None or pct_b is None:
                continue

            if not position:
                bb_condition = (pct_b <= 0.05) if require_bb_touch else True
                if rsi < rsi_buy and bb_condition:
                    trades.append({
                        "date": bar["date"], "action": "BUY",
                        "price": bar["close"], "rsi": round(rsi, 2), "bb_pct_b": pct_b,
                    })
                    position = True
            else:
                if rsi > rsi_sell or pct_b >= 0.5:
                    trades.append({
                        "date": bar["date"], "action": "SELL",
                        "price": bar["close"], "rsi": round(rsi, 2), "bb_pct_b": pct_b,
                    })
                    position = False

        return trades

    def macd_momentum_strategy(
        self,
        data: List[Dict],
        require_volume_spike: bool = True,
        volume_ratio_threshold: float = 1.2,
    ) -> List[Dict]:
        """
        MACD Momentum:
          BUY  when MACD histogram turns positive (prev ≤ 0, curr > 0)
               AND optionally volume_ratio > threshold (institutional participation).
          SELL when MACD histogram turns negative (prev ≥ 0, curr < 0).
        """
        enriched = compute_all(data)
        trades: List[Dict] = []
        position = False

        for i in range(1, len(enriched)):
            bar = enriched[i]
            prev = enriched[i - 1]
            hist = bar.get("macd_histogram")
            prev_hist = prev.get("macd_histogram")
            vol_ratio = bar.get("volume_ratio")

            if hist is None or prev_hist is None:
                continue

            vol_ok = (not require_volume_spike) or (vol_ratio is not None and vol_ratio >= volume_ratio_threshold)

            if not position:
                if prev_hist <= 0 and hist > 0 and vol_ok:
                    trades.append({
                        "date": bar["date"], "action": "BUY",
                        "price": bar["close"],
                        "macd_histogram": round(hist, 6),
                        "volume_ratio": round(vol_ratio, 4) if vol_ratio else None,
                    })
                    position = True
            else:
                if prev_hist >= 0 and hist < 0:
                    trades.append({
                        "date": bar["date"], "action": "SELL",
                        "price": bar["close"],
                        "macd_histogram": round(hist, 6),
                        "volume_ratio": round(vol_ratio, 4) if vol_ratio else None,
                    })
                    position = False

        return trades

    def combined_signal_strategy(
        self,
        data: List[Dict],
        buy_threshold: float = 0.5,
        sell_threshold: float = -0.3,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        """
        Composite score strategy: aggregates RSI, MACD histogram, Bollinger %B,
        and Volume Ratio into a single score from -1 to +1.

        Score > buy_threshold  → BUY
        Score < sell_threshold → SELL (or exit long)

        Default weights: rsi=0.3, macd=0.3, bb=0.2, volume=0.2
        """
        if weights is None:
            weights = {"rsi": 0.3, "macd": 0.3, "bb": 0.2, "volume": 0.2}

        enriched = compute_all(data)
        trades: List[Dict] = []
        position = False

        for i, bar in enumerate(enriched):
            rsi = bar.get("rsi_14")
            hist = bar.get("macd_histogram")
            pct_b = bar.get("bb_pct_b")
            vol_ratio = bar.get("volume_ratio")

            if rsi is None or hist is None or pct_b is None:
                continue

            # Normalise each indicator to [-1, +1]
            rsi_score = (rsi - 50.0) / 50.0          # 0→-1, 50→0, 100→+1
            # MACD histogram: cap at ±2 for normalisation
            hist_max = 2.0
            macd_score = max(-1.0, min(1.0, hist / hist_max)) if hist_max else 0.0
            bb_score = (pct_b - 0.5) * 2.0           # 0→-1, 0.5→0, 1→+1
            vol_score = min(1.0, (vol_ratio - 1.0)) if vol_ratio and vol_ratio > 0 else 0.0

            score = (
                weights.get("rsi", 0.3) * rsi_score
                + weights.get("macd", 0.3) * macd_score
                + weights.get("bb", 0.2) * bb_score
                + weights.get("volume", 0.2) * vol_score
            )

            if not position and score > buy_threshold:
                trades.append({
                    "date": bar["date"], "action": "BUY",
                    "price": bar["close"], "signal_score": round(score, 4),
                })
                position = True
            elif position and score < sell_threshold:
                trades.append({
                    "date": bar["date"], "action": "SELL",
                    "price": bar["close"], "signal_score": round(score, 4),
                })
                position = False

        return trades

    # ------------------------------------------------------------------
    # Regime-filtered backtest
    # ------------------------------------------------------------------

    def regime_filtered_backtest(
        self,
        strategy_name: str = "combined_signal",
        spy_closes: Optional[List[float]] = None,
        vix_close: Optional[float] = None,
        short_period: int = 10,
        long_period: int = 30,
        initial_capital: float = 10000.0,
        commission_per_trade: float = 1.0,
        slippage_pct: float = 0.0005,
        allow_short: bool = False,
    ) -> Dict:
        """
        Backtest with regime gating: trades only execute when the regime
        permits them (e.g. skip longs during a bear macro regime).

        When spy_closes is empty, macro regime defaults to NEUTRAL
        (useful for offline backtests without a live IBKR connection).
        """
        spy = spy_closes or []
        closes = [d["close"] for d in self.train_data]
        highs = [d["high"] for d in self.train_data]
        lows = [d["low"] for d in self.train_data]

        regime = get_regime(
            stock_closes=closes,
            spy_closes=spy,
            vix_close=vix_close,
            stock_highs=highs,
            stock_lows=lows,
            allow_short=allow_short,
        )

        # Pick buy threshold based on macro confidence
        threshold_map = {
            "bull": 0.40,
            "neutral": 0.55,
            "caution": 0.70,
            "bear": 0.90,  # Very hard to trigger a long in bear market
        }
        buy_threshold = threshold_map.get(regime.macro, 0.55)

        # Select strategy
        if strategy_name == "combined_signal":
            train_trades = self.combined_signal_strategy(
                self.train_data, buy_threshold=buy_threshold
            )
            val_trades = self.combined_signal_strategy(
                self.validation_data, buy_threshold=buy_threshold
            )
        elif strategy_name == "rsi_mean_reversion":
            train_trades = self.rsi_mean_reversion_strategy(self.train_data)
            val_trades = self.rsi_mean_reversion_strategy(self.validation_data)
        elif strategy_name == "macd_momentum":
            train_trades = self.macd_momentum_strategy(self.train_data)
            val_trades = self.macd_momentum_strategy(self.validation_data)
        else:
            train_trades = self.moving_average_crossover_strategy(
                self.train_data, short_period, long_period, allow_short
            )
            val_trades = self.moving_average_crossover_strategy(
                self.validation_data, short_period, long_period, allow_short
            )

        # Filter trades: remove longs if regime says skip, shorts if not allowed
        def _filter(trades: List[Dict]) -> List[Dict]:
            if regime.trade_direction == "none":
                return []
            if regime.trade_direction == "long":
                return [t for t in trades if t["action"] not in ("SHORT", "COVER")]
            return trades  # short direction: keep all

        train_trades = _filter(train_trades)
        val_trades = _filter(val_trades)

        train_perf = self.calculate_performance(
            train_trades, initial_capital, commission_per_trade, slippage_pct
        )
        val_perf = self.calculate_performance(
            val_trades, initial_capital, commission_per_trade, slippage_pct
        )

        return {
            "strategy": strategy_name,
            "regime": {
                "macro": regime.macro,
                "micro": regime.micro,
                "confidence": regime.confidence,
                "trade_direction": regime.trade_direction,
                "size_multiplier": regime.position_size_multiplier,
                "buy_threshold_used": buy_threshold,
            },
            "data_summary": self.get_data_summary(),
            "training_results": {"trades": train_trades, "performance": train_perf},
            "validation_results": {"trades": val_trades, "performance": val_perf},
            "summary": {
                "train_return": train_perf["total_return"],
                "validation_return": val_perf["total_return"],
                "performance_difference": round(
                    train_perf["total_return"] - val_perf["total_return"], 2
                ),
                "train_sharpe": train_perf["sharpe_ratio"],
                "validation_sharpe": val_perf["sharpe_ratio"],
            },
        }
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


@router.post("/regime_backtest/{file_id}")
async def regime_backtest(
    file_id: str = Path(..., description="File ID of saved historical data"),
    strategy: str = Query(
        "combined_signal",
        description="Strategy: combined_signal | rsi_mean_reversion | macd_momentum | moving_average_crossover",
    ),
    train_split: float = Query(0.8, description="Training split fraction"),
    initial_capital: float = Query(10000.0, description="Initial capital"),
    commission: float = Query(1.0, description="Commission per trade"),
    slippage: float = Query(0.0005, description="Slippage fraction"),
    allow_short: bool = Query(False, description="Allow short selling"),
    output_dir: str = Query("historical_data", description="Directory with JSON files"),
):
    """
    Backtest with regime-aware gating.

    Classifies the current market regime from the training window and adjusts
    the signal entry threshold accordingly.  Trades that conflict with the
    detected regime are filtered out before performance is measured.

    Pass spy_closes via JSON body for a live macro regime; leave empty to use
    the symbol's own data as a regime proxy (useful for offline backtests).
    """

    def execute():
        if not os.path.exists(output_dir):
            return {"error": "Historical data directory not found"}

        matching = [f for f in os.listdir(output_dir) if f.endswith(f"_{file_id}.json")]
        if not matching:
            return {"error": f"No file found with ID {file_id}"}

        with open(os.path.join(output_dir, matching[0]), "r") as fh:
            file_data = json.load(fh)

        data = file_data.get("data", [])
        if not data:
            return {"error": "No data found in file"}

        engine = BacktestEngine(data, train_split=train_split)
        results = engine.regime_filtered_backtest(
            strategy_name=strategy,
            spy_closes=[],   # No live SPY; macro inferred from symbol's own SMA
            vix_close=None,
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

    result = await run_in_threadpool(execute)
    return JSONResponse(result)


@router.post("/oos_backtest/{file_id}")
async def oos_backtest_file(
    file_id: str = Path(..., description="File ID from historical_data/ (use a 5-10Y daily file)"),
    is_pct: float = Query(0.6, description="In-Sample fraction, e.g. 0.6 = first 60% for training"),
    initial_capital: float = Query(10000.0),
    commission: float = Query(1.0),
    slippage: float = Query(0.0005),
    save_graph: bool = Query(True, description="Save equity curve PNG to graphs/"),
    output_dir: str = Query("historical_data", description="Directory with JSON files"),
):
    """
    Walk-Forward Holdout Backtest on a saved JSON file.

    Splits data into IS (training) and OOS (blind simulation) windows,
    runs 7 proven swing-trading strategies, optimises each on IS only,
    then simulates the OOS period as if live.

    Returns per-strategy IS + OOS metrics, a formatted comparison table,
    and optionally saves an equity-curve PNG graph.

    Recommended: use a 5-10 year daily file for meaningful results.
    """
    def execute():
        if not os.path.exists(output_dir):
            return {"error": "Historical data directory not found"}
        matching = [f for f in os.listdir(output_dir) if f.endswith(f"_{file_id}.json")]
        if not matching:
            return {"error": f"No file found with ID {file_id}"}

        with open(os.path.join(output_dir, matching[0]), "r") as fh:
            file_data = json.load(fh)
        data = file_data.get("data", [])
        if not data:
            return {"error": "No data found in file"}
        if len(data) < 200:
            return {"error": f"Need at least 200 bars for OOS backtest (got {len(data)}). "
                            "Fetch 5-10 years of daily data first."}

        from oos_backtest import OOSBacktest
        symbol = file_data.get("symbol", "")
        engine = OOSBacktest(data, is_pct=is_pct,
                             initial_capital=initial_capital,
                             commission=commission, slippage=slippage)
        results = engine.run()

        if save_graph:
            os.makedirs("graphs", exist_ok=True)
            graph_path = f"graphs/oos_{symbol.lower()}_{file_id}.png"
            engine.generate_graph(results, graph_path, symbol=symbol)
            results["graph_saved_to"] = graph_path

        results["file_info"] = {
            "file_id": file_id,
            "symbol": symbol,
            "duration": file_data.get("duration"),
            "bar_size": file_data.get("bar_size"),
        }
        # Remove large equity arrays + raw trades from JSON to keep response lean
        for strat_res in results.get("strategies", {}).values():
            strat_res.pop("oos_equity_dates", None)
            strat_res.pop("oos_equity_values", None)
            strat_res.pop("oos_equity_daily", None)
            strat_res.pop("oos_trades", None)
        results.get("strategies", {}).get("buy_and_hold", {}).pop("oos_equity_dates", None)
        results.get("strategies", {}).get("buy_and_hold", {}).pop("oos_equity_values", None)
        results.get("strategies", {}).get("buy_and_hold", {}).pop("oos_equity_daily", None)
        results.get("spy_benchmark", {}).pop("dates", None)
        results.get("spy_benchmark", {}).pop("equity", None)
        return results

    result = await run_in_threadpool(execute)
    return JSONResponse(result)


@router.post("/oos_backtest_cached/{symbol}")
async def oos_backtest_cached(
    symbol: str = Path(..., description="Stock symbol (uses data_cache/ for 5Y daily bars)"),
    is_pct: float = Query(0.6, description="In-Sample fraction"),
    initial_capital: float = Query(10000.0),
    commission: float = Query(1.0),
    slippage: float = Query(0.0005),
    save_graph: bool = Query(True),
):
    """
    Walk-Forward Holdout Backtest using the data cache (5Y daily bars).

    If the cache is empty for this symbol, it fetches 5Y of data from IBKR first
    (this may take a moment on the first call).

    Same output as /oos_backtest/{file_id} but uses the auto-managed cache
    rather than a manually saved file.
    """
    def execute():
        from data_cache import DataCache
        from main import ib_client
        cache = DataCache(ib_client)
        data = cache.get(symbol.upper(), "1 day")

        if not data:
            return {"error": f"No cached data for {symbol}. "
                            "Call POST /bot/warm_cache first, or ensure IBKR is connected."}
        if len(data) < 200:
            return {"error": f"Only {len(data)} bars cached — need ≥200 for OOS backtest."}

        from oos_backtest import OOSBacktest
        engine = OOSBacktest(data, is_pct=is_pct,
                             initial_capital=initial_capital,
                             commission=commission, slippage=slippage)
        results = engine.run()

        if save_graph:
            os.makedirs("graphs", exist_ok=True)
            graph_path = f"graphs/oos_{symbol.lower()}_cached.png"
            engine.generate_graph(results, graph_path, symbol=symbol)
            results["graph_saved_to"] = graph_path

        results["source"] = f"cache/1_day ({len(data)} bars)"
        for strat_res in results.get("strategies", {}).values():
            strat_res.pop("oos_equity_dates", None)
            strat_res.pop("oos_equity_values", None)
            strat_res.pop("oos_equity_daily", None)
            strat_res.pop("oos_trades", None)
        results.get("strategies", {}).get("buy_and_hold", {}).pop("oos_equity_dates", None)
        results.get("strategies", {}).get("buy_and_hold", {}).pop("oos_equity_values", None)
        results.get("strategies", {}).get("buy_and_hold", {}).pop("oos_equity_daily", None)
        results.get("spy_benchmark", {}).pop("dates", None)
        results.get("spy_benchmark", {}).pop("equity", None)
        return results

    result = await run_in_threadpool(execute)
    return JSONResponse(result)