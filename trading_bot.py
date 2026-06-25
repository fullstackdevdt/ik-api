"""
Trading bot orchestrator.

Scans a watchlist on a schedule (via APScheduler, 30-min cycles),
evaluates macro+micro regime, sizes positions with ATR stops,
and routes paper orders through IBKRClient (port 4002).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

from data_cache import DataCache
from indicators import compute_all
from regime import get_regime
from risk_manager import (
    PortfolioLimits,
    calculate_atr_stop,
    calculate_position_size,
    calculate_profit_target,
    can_open_new_position,
    check_circuit_breaker,
)
from portfolio_state import (
    BotConfig,
    ClosedTrade,
    Position,
    PortfolioState,
    get_performance_summary,
    load_state,
    save_state,
)

logger = logging.getLogger(__name__)


class BotEngine:
    """
    Core orchestrator: scan → regime → signal → size → order → record.
    Instantiated once per bot session; state is persisted to JSON after
    every cycle so restarts are safe.
    """

    def __init__(self, ib_client) -> None:
        self.ib_client = ib_client
        self.state: PortfolioState = load_state()
        self._cache = DataCache(ib_client)

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _fetch_daily_data(self, symbol: str) -> List[Dict]:
        """
        Up to 5 years of daily OHLCV bars (cache/tsla_1_day.json).
        First call fetches full history from IBKR; subsequent calls only
        pull the delta since the last cached bar.
        """
        return self._cache.get(symbol, "1 day")

    def _fetch_4h_data(self, symbol: str) -> List[Dict]:
        """
        Up to 1 year of 4-hour OHLCV bars (cache/tsla_4_hours.json).
        First call fetches full history; subsequent calls pull delta only.
        """
        return self._cache.get(symbol, "4 hours")

    def _fetch_spy_vix_context(self) -> Dict:
        """Cached SPY daily closes (5Y) + latest VIX close for macro regime."""
        return self._cache.get_spy_vix()

    # ------------------------------------------------------------------
    # 4-hour confirmation gate
    # ------------------------------------------------------------------

    def _confirm_entry_on_4h(self, data_4h: List[Dict], direction: str) -> bool:
        """
        Require the most recent 4-hour bars to show the same directional
        bias as the daily signal.  Falls back to True when data is sparse.
        """
        if not data_4h or len(data_4h) < 20:
            return True  # Insufficient 4h data — skip filter

        enriched = compute_all(data_4h)
        last = enriched[-1]
        prev = enriched[-2] if len(enriched) >= 2 else None

        rsi = last.get("rsi_14")
        hist_now = last.get("macd_histogram")
        hist_prev = prev.get("macd_histogram") if prev else None

        if direction == "long":
            rsi_ok = rsi is not None and rsi > 40
            macd_ok = (
                hist_now is not None
                and hist_prev is not None
                and hist_now >= hist_prev
            )
        else:  # short
            rsi_ok = rsi is not None and rsi < 60
            macd_ok = (
                hist_now is not None
                and hist_prev is not None
                and hist_now <= hist_prev
            )

        return bool(rsi_ok or macd_ok)

    # ------------------------------------------------------------------
    # Scan pipeline
    # ------------------------------------------------------------------

    def scan_symbol(
        self,
        symbol: str,
        spy_closes: List[float],
        vix_close: Optional[float],
    ) -> Dict:
        """
        Full analysis for one symbol.
        Returns a signal dict with regime classification and trade recommendation.
        """
        daily_data = self._fetch_daily_data(symbol)
        if not daily_data:
            return {"symbol": symbol, "error": "No daily data returned"}

        enriched = compute_all(daily_data)
        last_bar = enriched[-1]

        closes = [d["close"] for d in daily_data]
        highs = [d["high"] for d in daily_data]
        lows = [d["low"] for d in daily_data]

        regime = get_regime(
            stock_closes=closes,
            spy_closes=spy_closes,
            vix_close=vix_close,
            stock_highs=highs,
            stock_lows=lows,
            allow_short=self.state.config.allow_short,
        )

        # Only fetch 4h bars if regime suggests an actionable trade
        confirmed_4h = True
        if regime.trade_direction != "none" and regime.position_size_multiplier > 0:
            try:
                data_4h = self._fetch_4h_data(symbol)
                confirmed_4h = self._confirm_entry_on_4h(data_4h, regime.trade_direction)
            except Exception as e:
                logger.warning(f"4h fetch failed for {symbol}: {e}")

        final_action = regime.trade_direction if confirmed_4h else "none"
        final_mult = regime.position_size_multiplier if confirmed_4h else 0.0

        return {
            "symbol": symbol,
            "date": last_bar.get("date"),
            "close": last_bar.get("close"),
            "rsi_14": last_bar.get("rsi_14"),
            "macd_histogram": last_bar.get("macd_histogram"),
            "atr_14": last_bar.get("atr_14"),
            "atr_pct": last_bar.get("atr_pct"),
            "bb_pct_b": last_bar.get("bb_pct_b"),
            "adx_14": last_bar.get("adx_14"),
            "volume_ratio": last_bar.get("volume_ratio"),
            "regime": {
                "macro": regime.macro,
                "micro": regime.micro,
                "confidence": regime.confidence,
                "trade_direction": regime.trade_direction,
                "size_multiplier": regime.position_size_multiplier,
            },
            "4h_confirmed": confirmed_4h,
            "action": final_action,
            "size_multiplier": final_mult,
        }

    # ------------------------------------------------------------------
    # Entry / exit decisions
    # ------------------------------------------------------------------

    def evaluate_entry(self, scan_result: Dict) -> Optional[Dict]:
        """
        Turn a scan result into an order spec, or None if no trade.
        Checks: action != none, no existing position, portfolio capacity.
        """
        symbol = scan_result.get("symbol", "")
        action = scan_result.get("action", "none")
        size_mult = scan_result.get("size_multiplier", 0.0)

        if action == "none" or size_mult <= 0:
            return None
        if symbol in self.state.positions:
            return None  # Already in a position
        if not can_open_new_position(
            len(self.state.positions),
            PortfolioLimits(max_positions=self.state.config.max_positions),
        ):
            return None

        entry_price = scan_result.get("close")
        atr = scan_result.get("atr_14")
        if not entry_price or not atr:
            return None

        direction = "long" if action == "long" else "short"
        stop = calculate_atr_stop(
            entry_price, atr,
            multiplier=self.state.config.atr_stop_multiplier,
            direction=direction,
        )
        target = calculate_profit_target(
            entry_price, stop,
            reward_risk_ratio=self.state.config.profit_target_rr,
            direction=direction,
        )
        limits = PortfolioLimits(
            max_positions=self.state.config.max_positions,
            max_risk_pct_per_trade=self.state.config.risk_pct,
        )
        spec = calculate_position_size(
            capital=self.state.capital,
            entry_price=entry_price,
            stop_price=stop,
            profit_target=target,
            risk_pct=self.state.config.risk_pct,
            size_multiplier=size_mult,
            symbol=symbol,
            limits=limits,
        )

        if spec.shares <= 0:
            return None

        return {
            "symbol": symbol,
            "direction": direction,
            "ib_action": "BUY" if direction == "long" else "SELL",
            "shares": spec.shares,
            "entry_price": entry_price,
            "stop_price": stop,
            "profit_target": target,
            "dollar_risk": spec.dollar_risk,
            "position_value": spec.position_value,
            "atr": atr,
            "size_multiplier": size_mult,
        }

    def evaluate_exits(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Check all open positions for stop-loss, profit-target, or time-exit triggers.
        Returns a list of exit order dicts.
        """
        exits = []
        for symbol, pos in list(self.state.positions.items()):
            price = current_prices.get(symbol, pos.current_price)
            if price <= 0:
                continue

            reason: Optional[str] = None

            if pos.direction == "long":
                if price <= pos.stop_price:
                    reason = "stop_loss"
                elif price >= pos.profit_target:
                    reason = "profit_target"
            else:
                if price >= pos.stop_price:
                    reason = "stop_loss"
                elif price <= pos.profit_target:
                    reason = "profit_target"

            # Time-based exit
            if reason is None:
                try:
                    entry_dt = datetime.strptime(pos.entry_date, "%Y-%m-%d")
                    hold_days = (datetime.now() - entry_dt).days
                    if hold_days >= self.state.config.max_hold_days:
                        reason = "time_exit"
                except Exception:
                    pass

            if reason:
                exits.append({
                    "symbol": symbol,
                    "ib_action": "SELL" if pos.direction == "long" else "BUY",
                    "shares": pos.shares,
                    "current_price": price,
                    "exit_reason": reason,
                    "position": pos,
                })

        return exits

    # ------------------------------------------------------------------
    # State recording
    # ------------------------------------------------------------------

    def record_entry(self, order: Dict, strategy: str = "regime_multi_signal") -> None:
        """Persist a newly executed entry into portfolio state."""
        pos = Position(
            symbol=order["symbol"],
            direction=order["direction"],
            entry_price=order["entry_price"],
            shares=order["shares"],
            stop_price=order["stop_price"],
            profit_target=order["profit_target"],
            entry_date=datetime.now().strftime("%Y-%m-%d"),
            strategy=strategy,
            atr_at_entry=order.get("atr", 0.0),
            size_multiplier=order.get("size_multiplier", 1.0),
            current_price=order["entry_price"],
        )
        self.state.positions[order["symbol"]] = pos
        self.state.capital -= order["position_value"]
        self.state.peak_capital = max(self.state.peak_capital, self.state.capital)
        save_state(self.state)

    def record_exit(self, exit_order: Dict) -> None:
        """Persist a closed trade and return capital to the cash balance."""
        pos: Position = exit_order["position"]
        price = exit_order["current_price"]
        symbol = exit_order["symbol"]

        pnl = (
            (price - pos.entry_price) * pos.shares
            if pos.direction == "long"
            else (pos.entry_price - price) * pos.shares
        )
        pnl_pct = pnl / (pos.entry_price * pos.shares) * 100 if pos.shares > 0 else 0.0

        hold_days = 0
        try:
            hold_days = (
                datetime.now() - datetime.strptime(pos.entry_date, "%Y-%m-%d")
            ).days
        except Exception:
            pass

        self.state.closed_trades.append(ClosedTrade(
            symbol=symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=price,
            shares=pos.shares,
            entry_date=pos.entry_date,
            exit_date=datetime.now().strftime("%Y-%m-%d"),
            exit_reason=exit_order["exit_reason"],
            pnl_dollars=round(pnl, 2),
            pnl_pct=round(pnl_pct, 3),
            strategy=pos.strategy,
            hold_days=hold_days,
        ))

        del self.state.positions[symbol]
        self.state.capital += pos.entry_price * pos.shares + pnl  # Return basis + P&L
        self.state.peak_capital = max(self.state.peak_capital, self.state.capital)
        save_state(self.state)
        logger.info(
            f"EXIT {symbol} ({exit_order['exit_reason']}): "
            f"P&L ${pnl:.2f} ({pnl_pct:.2f}%)"
        )

    # ------------------------------------------------------------------
    # Main scan cycle (called by APScheduler every 30 min)
    # ------------------------------------------------------------------

    def run_cycle(self) -> Dict:
        """
        Full scan cycle:
          1. Check circuit breaker
          2. Fetch macro context (SPY + VIX)
          3. Evaluate exits on open positions
          4. Scan watchlist for new entries
          5. Execute paper orders via IBKR
        """
        if self.state.bot_status != "running":
            return {"status": self.state.bot_status, "message": "Bot is not running"}

        now = datetime.now()
        self.state.last_scan = now.strftime("%Y-%m-%d %H:%M:%S")

        # Compute portfolio P&L for circuit breaker check
        position_value = sum(
            p.current_price * p.shares for p in self.state.positions.values()
        )
        total_capital = self.state.capital + position_value
        daily_pnl_pct = (
            (total_capital - self.state.start_of_day_capital) / self.state.start_of_day_capital
            if self.state.start_of_day_capital > 0 else 0.0
        )
        weekly_pnl_pct = (
            (total_capital - self.state.start_of_week_capital) / self.state.start_of_week_capital
            if self.state.start_of_week_capital > 0 else 0.0
        )
        drawdown_pct = (
            (self.state.peak_capital - total_capital) / self.state.peak_capital
            if self.state.peak_capital > 0 else 0.0
        )

        cb = check_circuit_breaker(daily_pnl_pct, weekly_pnl_pct, -drawdown_pct)
        if cb["halt"]:
            self.state.bot_status = "circuit_breaker"
            self.state.circuit_breaker_reason = "; ".join(cb["reasons"])
            save_state(self.state)
            logger.warning(f"Circuit breaker: {self.state.circuit_breaker_reason}")
            return {"status": "circuit_breaker", "reasons": cb["reasons"]}

        # Macro context (shared across all symbols this cycle)
        spy_closes: List[float] = []
        vix_close: Optional[float] = None
        try:
            ctx = self._fetch_spy_vix_context()
            spy_closes = ctx.get("spy_closes", [])
            vix_close = ctx.get("vix_close")
        except Exception as e:
            logger.error(f"Macro context fetch failed: {e}")

        # ------ Exits first ------
        current_prices: Dict[str, float] = {}
        for sym in list(self.state.positions.keys()):
            try:
                current_prices[sym] = self.ib_client.get_realtime_price(sym)
            except Exception as e:
                logger.warning(f"Price fetch failed for {sym}: {e}")

        exit_orders = self.evaluate_exits(current_prices)
        executed_exits = 0
        for ex in exit_orders:
            try:
                from ib_insync import MarketOrder, Stock
                contract = Stock(ex["symbol"], "SMART", "USD")
                self.ib_client.place_market_order(contract, MarketOrder(ex["ib_action"], ex["shares"]))
                self.record_exit(ex)
                executed_exits += 1
            except Exception as e:
                logger.error(f"Exit order failed ({ex['symbol']}): {e}")

        # ------ Entries ------
        scan_results = []
        executed_entries = 0
        for sym in self.state.config.watchlist:
            if sym in self.state.positions:
                continue
            try:
                result = self.scan_symbol(sym, spy_closes, vix_close)
                scan_results.append(result)
                order = self.evaluate_entry(result)
                if order:
                    from ib_insync import MarketOrder, Stock
                    contract = Stock(sym, "SMART", "USD")
                    self.ib_client.place_market_order(
                        contract, MarketOrder(order["ib_action"], order["shares"])
                    )
                    self.record_entry(order)
                    executed_entries += 1
                    logger.info(
                        f"ENTRY {sym}: {order['direction']} {order['shares']} shares "
                        f"@ {order['entry_price']}, stop={order['stop_price']}"
                    )
            except Exception as e:
                logger.error(f"Scan/entry failed ({sym}): {e}")
                scan_results.append({"symbol": sym, "error": str(e)})

        save_state(self.state)
        return {
            "status": "ok",
            "scan_time": self.state.last_scan,
            "symbols_scanned": len(scan_results),
            "executed_entries": executed_entries,
            "executed_exits": executed_exits,
            "open_positions": len(self.state.positions),
            "available_cash": round(self.state.capital, 2),
            "scan_results": scan_results,
        }
