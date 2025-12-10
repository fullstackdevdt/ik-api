from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Iterable, List, Optional
from datetime import datetime
from ib_insync import IB, Stock, Contract, BarDataList, Order, Trade
import logging
import matplotlib.pyplot as plt
import os
import time
from threading import Lock

logger = logging.getLogger(__name__)

class IBKRConnectionError(Exception):
    """Raised when unable to connect to IBKR."""
    pass

class IBKRClient:
    """Thin wrapper around ib_insync with shared connection + helpers."""

    def __init__(self, host: str, port: int, client_id: int = 1, rate_limit_delay: float = 0.1) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.rate_limit_delay = rate_limit_delay  # Minimum seconds between requests
        self._last_request_time = 0.0
        self._rate_limit_lock = Lock()

    def _enforce_rate_limit(self) -> None:
        """Ensures minimum delay between API requests."""
        with self._rate_limit_lock:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
            self._last_request_time = time.time()

    def _ensure_loop(self) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    @contextmanager
    def connection(self) -> Iterable[IB]:
        self._enforce_rate_limit()  # Add rate limiting before each connection
        self._ensure_loop()
        ib = IB()
        try:
            ib.connect(self.host, self.port, clientId=self.client_id)
            yield ib
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            raise IBKRConnectionError(f"Unable to connect to IBKR at {self.host}:{self.port}") from e
        finally:
            if ib.isConnected():
                ib.disconnect()

    def get_realtime_price(self, symbol: str, market_data_type: int = 3) -> float:
        with self.connection() as ib:
            ib.reqMarketDataType(market_data_type)
            contract = Stock(symbol.upper(), "SMART", "USD")
            ticker = ib.reqMktData(contract, "", False, False)
            ib.sleep(2)
            return ticker.marketPrice()

    def get_historical_data(
        self,
        symbol: str,
        duration: str,
        bar_size: str,
        what_to_show: str,
        end_time: str = "",
        use_rth: bool = True,
    ) -> BarDataList:
        with self.connection() as ib:
            contract = Stock(symbol.upper(), "SMART", "USD")
            return ib.reqHistoricalData(
                contract,
                endDateTime=end_time,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1,
            )

    def place_market_order(
        self, contract: Contract, order: Order, sleep_seconds: float = 2.0
    ) -> Trade:
        with self.connection() as ib:
            ib.reqMarketDataType(3)
            trade = ib.placeOrder(contract, order)
            ib.sleep(sleep_seconds)
            return trade

    def get_account_summary(self) -> List:
        with self.connection() as ib:
            return ib.accountSummary()
        
    def get_head_timestamp(self, symbol: str, what_to_show: str = 'TRADES') -> Optional[datetime]:
        with self.connection() as ib:
            contract = Stock(symbol.upper(), "SMART", "USD")
            return ib.reqHeadTimeStamp(contract, whatToShow=what_to_show, useRTH=True)
        

    def generate_historical_graph(
        self,
        symbol: str,
        duration: str = "1 M",
        bar_size: str = "1 day",
        output_dir: str = "graphs",
    ) -> Optional[str]:
        """
        Fetches historical data and generates a graph, saving it as a PNG file.

        Args:
            symbol: The stock symbol.
            duration: The duration to fetch data for (e.g., '1 M', '1 Y').
            bar_size: The bar size for the data (e.g., '1 day', '1 hour').
            output_dir: The directory to save the graph in.

        Returns:
            The file path of the generated graph, or None if no data was found.
        """
        bars = self.get_historical_data(
            symbol=symbol,
            duration=duration,
            bar_size=bar_size,
            what_to_show="TRADES",
        )

        if not bars:
            logger.warning(f"No historical data found for {symbol}")
            return None

        dates = [bar.date for bar in bars]
        closes = [bar.close for bar in bars]

        plt.figure(figsize=(10, 5))
        plt.plot(dates, closes, marker="o")
        plt.title(f"{symbol.upper()} Historical Close Prices ({duration})")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = f"{symbol.lower()}_{duration.replace(' ', '')}_{bar_size.replace(' ', '')}_historical.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()

        logger.info(f"Generated historical graph for {symbol} at {filepath}")
        return filepath
    

    def simulate_swing_trade(
        self,
        symbol: str,
        duration: str = "1 Y",
        bar_size: str = "1 day",
        short_ma: int = 10,
        long_ma: int = 30,
        output_dir: str = "graphs",
    ) -> Optional[str]:
        """
        Simulates swing trading using Moving Average Crossover strategy.
        Uses ONLY the first 50% of historical data for backtesting.
        
        Strategy: Buy when short MA crosses above long MA, sell when it crosses below.
        
        Returns: Path to generated graph showing:
        - Blue line: Full price data (paper trading reference - all data)
        - Green triangles: Buy signals (on backtest portion)
        - Red triangles: Sell signals (on backtest portion)
        - Red line segments: Periods when holding stock (swing trades on backtest portion)
        """
        
        # Step 1: Get historical data
        bars = self.get_historical_data(
            symbol=symbol,
            duration=duration,
            bar_size=bar_size,
            what_to_show="TRADES",
        )

        if not bars:
            logger.warning(f"No historical data found for {symbol}")
            return None

        # Step 2: Split data - first 50% for backtest, full data for reference
        half_point = len(bars) // 2
        backtest_data = bars[:half_point]
        
        # Full dataset for blue line
        all_dates = [bar.date for bar in bars]
        all_prices = [bar.close for bar in bars]
        
        if len(backtest_data) < long_ma:
            logger.warning(f"Need at least {long_ma} data points, got {len(backtest_data)}")
            return None

        # Step 3: Extract backtest price data
        dates = [bar.date for bar in backtest_data]
        prices = [bar.close for bar in backtest_data]

        # Step 4: Calculate moving averages
        short_ma_values = []
        long_ma_values = []
        
        for i in range(len(prices)):
            if i >= short_ma - 1:
                short_avg = sum(prices[i-short_ma+1:i+1]) / short_ma
                short_ma_values.append(short_avg)
            else:
                short_ma_values.append(None)
                
            if i >= long_ma - 1:
                long_avg = sum(prices[i-long_ma+1:i+1]) / long_ma
                long_ma_values.append(long_avg)
            else:
                long_ma_values.append(None)

        # Step 5: Generate buy/sell signals
        buy_signals = []  # [(date, price), ...]
        sell_signals = []
        position_open = False
        
        for i in range(1, len(prices)):
            curr_short = short_ma_values[i]
            curr_long = long_ma_values[i]
            prev_short = short_ma_values[i-1]
            prev_long = long_ma_values[i-1]
            
            if curr_short is None or curr_long is None:
                continue
            if prev_short is None or prev_long is None:
                continue
            
            # Buy signal: short MA crosses above long MA
            if prev_short <= prev_long and curr_short > curr_long and not position_open:
                buy_signals.append((dates[i], prices[i]))
                position_open = True
                
            # Sell signal: short MA crosses below long MA  
            elif prev_short >= prev_long and curr_short < curr_long and position_open:
                sell_signals.append((dates[i], prices[i]))
                position_open = False

        # Step 6: Create the graph
        plt.figure(figsize=(14, 8))
        
        # Plot FULL price line in blue (paper trading - entire dataset)
        plt.plot(all_dates, all_prices, color='blue', alpha=0.5, 
                label=f'{symbol} Full Price Data (Paper Trade Reference)', linewidth=2)
        
        # Plot buy signals (green triangles) - only on backtest portion
        if buy_signals:
            buy_dates, buy_prices = zip(*buy_signals)
            plt.scatter(buy_dates, buy_prices, marker='^', color='green', 
                       s=150, label='Buy Signal', zorder=5, edgecolors='black', linewidths=1)
        
        # Plot sell signals (red triangles) - only on backtest portion
        if sell_signals:
            sell_dates, sell_prices = zip(*sell_signals)
            plt.scatter(sell_dates, sell_prices, marker='v', color='red', 
                       s=150, label='Sell Signal', zorder=5, edgecolors='black', linewidths=1)
        
        # Plot red line segments for holding periods (swing trades) - only on backtest portion
        trade_periods = []
        for i, (buy_date, _) in enumerate(buy_signals):
            if i < len(sell_signals):
                sell_date = sell_signals[i][0]
                trade_periods.append((buy_date, sell_date))
            else:
                # Open position at end of backtest
                trade_periods.append((buy_date, dates[-1]))
        
        for start_date, end_date in trade_periods:
            start_idx = dates.index(start_date)
            try:
                end_idx = dates.index(end_date)
            except ValueError:
                end_idx = len(dates) - 1
                
            plt.plot(dates[start_idx:end_idx+1], prices[start_idx:end_idx+1], 
                    color='red', linewidth=4, alpha=0.9, zorder=4)

        # Add first trade period to legend
        if trade_periods:
            plt.plot([], [], color='red', linewidth=4, alpha=0.9, 
                    label='Swing Trade Periods (Backtest)')

        # Add vertical line to show backtest cutoff
        if len(dates) > 0:
            plt.axvline(x=dates[-1], color='gray', linestyle='--', linewidth=2, 
                       alpha=0.7, label='Backtest End (50% of data)')

        plt.title(f'{symbol.upper()} Swing Trade Simulation\n'
                 f'Strategy: MA{short_ma}/{long_ma} Crossover | Backtest: First {len(backtest_data)} days | Full Dataset: {len(bars)} days')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Step 7: Save graph
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        filename = f"{symbol.lower()}_swing_simulation_MA{short_ma}_{long_ma}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        # Step 8: Log results
        total_trades = len(sell_signals)
        logger.info(f"Swing trade simulation completed for {symbol}")
        logger.info(f"Generated {total_trades} complete trades using first {len(backtest_data)} days")
        logger.info(f"Full dataset: {len(bars)} days, Backtest: {len(backtest_data)} days")
        logger.info(f"Graph saved to: {filepath}")
        
        return filepath