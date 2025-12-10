from fastapi import APIRouter, Path, Query
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime

router = APIRouter()


class BacktestEngine:
    """
    Engine for backtesting trading strategies on historical data.
    Splits data into training (80-90%) and validation (10-20%) periods.
    """
    
    def __init__(self, data: List[Dict], train_split: float = 0.8):
        """
        Initialize the backtest engine.
        
        Args:
            data: List of OHLCV data points from JSON file
            train_split: Percentage of data to use for training (0.8 = 80%)
        """
        self.full_data = data
        self.train_split = train_split
        
        # Split data into training and validation
        split_index = int(len(data) * train_split)
        self.train_data = data[:split_index]
        self.validation_data = data[split_index:]
        
    def get_data_summary(self) -> Dict:
        """Returns summary of the data split."""
        return {
            "total_data_points": len(self.full_data),
            "train_data_points": len(self.train_data),
            "validation_data_points": len(self.validation_data),
            "train_period": {
                "start": self.train_data[0]["date"] if self.train_data else None,
                "end": self.train_data[-1]["date"] if self.train_data else None
            },
            "validation_period": {
                "start": self.validation_data[0]["date"] if self.validation_data else None,
                "end": self.validation_data[-1]["date"] if self.validation_data else None
            },
            "train_split_percentage": self.train_split * 100
        }
    
    def moving_average_crossover_strategy(
        self,
        data: List[Dict],
        short_period: int = 10,
        long_period: int = 30
    ) -> List[Dict]:
        """
        Simple Moving Average Crossover Strategy.
        Buy when short MA crosses above long MA, sell when it crosses below.
        
        Returns list of trades: [{"date": str, "action": "BUY/SELL", "price": float}, ...]
        """
        if len(data) < long_period:
            return []
        
        trades = []
        position_open = False
        
        for i in range(long_period, len(data)):
            # Calculate short MA
            short_prices = [data[j]["close"] for j in range(i - short_period, i)]
            short_ma = sum(short_prices) / short_period
            
            # Calculate long MA
            long_prices = [data[j]["close"] for j in range(i - long_period, i)]
            long_ma = sum(long_prices) / long_period
            
            # Previous MAs
            prev_short_prices = [data[j]["close"] for j in range(i - short_period - 1, i - 1)]
            prev_short_ma = sum(prev_short_prices) / short_period
            
            prev_long_prices = [data[j]["close"] for j in range(i - long_period - 1, i - 1)]
            prev_long_ma = sum(prev_long_prices) / long_period
            
            # Check for crossover
            if prev_short_ma <= prev_long_ma and short_ma > long_ma and not position_open:
                # Buy signal
                trades.append({
                    "date": data[i]["date"],
                    "action": "BUY",
                    "price": data[i]["close"],
                    "short_ma": short_ma,
                    "long_ma": long_ma
                })
                position_open = True
                
            elif prev_short_ma >= prev_long_ma and short_ma < long_ma and position_open:
                # Sell signal
                trades.append({
                    "date": data[i]["date"],
                    "action": "SELL",
                    "price": data[i]["close"],
                    "short_ma": short_ma,
                    "long_ma": long_ma
                })
                position_open = False
        
        return trades
    
    def calculate_performance(self, trades: List[Dict], initial_capital: float = 10000.0) -> Dict:
        """
        Calculate performance metrics from a list of trades.
        
        Returns:
            Dict with metrics like total_return, win_rate, num_trades, etc.
        """
        if not trades or len(trades) < 2:
            return {
                "total_trades": 0,
                "profitable_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_return": 0.0,
                "final_capital": initial_capital
            }
        
        capital = initial_capital
        shares = 0
        profitable_trades = 0
        losing_trades = 0
        
        for trade in trades:
            if trade["action"] == "BUY":
                # Buy as many shares as possible with current capital
                shares = capital / trade["price"]
                capital = 0
            elif trade["action"] == "SELL" and shares > 0:
                # Sell all shares
                sell_value = shares * trade["price"]
                
                # Determine if trade was profitable
                if sell_value > initial_capital:
                    profitable_trades += 1
                else:
                    losing_trades += 1
                
                capital = sell_value
                shares = 0
        
        # If still holding shares, calculate final value
        if shares > 0 and trades:
            capital = shares * trades[-1]["price"]
        
        total_completed_trades = profitable_trades + losing_trades
        total_return = ((capital - initial_capital) / initial_capital) * 100
        
        return {
            "total_trades": len([t for t in trades if t["action"] == "BUY"]),
            "completed_trades": total_completed_trades,
            "profitable_trades": profitable_trades,
            "losing_trades": losing_trades,
            "win_rate": (profitable_trades / total_completed_trades * 100) if total_completed_trades > 0 else 0.0,
            "initial_capital": initial_capital,
            "final_capital": round(capital, 2),
            "total_return": round(total_return, 2),
            "total_return_dollars": round(capital - initial_capital, 2)
        }
    
    def run_backtest(
        self,
        strategy_name: str = "moving_average_crossover",
        short_period: int = 10,
        long_period: int = 30,
        initial_capital: float = 10000.0
    ) -> Dict:
        """
        Run backtest on training data and validate on validation data.
        
        Returns:
            Dict with training and validation results
        """
        # Run strategy on training data
        if strategy_name == "moving_average_crossover":
            train_trades = self.moving_average_crossover_strategy(
                self.train_data, short_period, long_period
            )
            validation_trades = self.moving_average_crossover_strategy(
                self.validation_data, short_period, long_period
            )
        else:
            return {"error": f"Unknown strategy: {strategy_name}"}
        
        # Calculate performance
        train_performance = self.calculate_performance(train_trades, initial_capital)
        validation_performance = self.calculate_performance(validation_trades, initial_capital)
        
        return {
            "strategy": strategy_name,
            "parameters": {
                "short_period": short_period,
                "long_period": long_period,
                "initial_capital": initial_capital
            },
            "data_summary": self.get_data_summary(),
            "training_results": {
                "trades": train_trades,
                "performance": train_performance
            },
            "validation_results": {
                "trades": validation_trades,
                "performance": validation_performance
            },
            "summary": {
                "train_return": train_performance["total_return"],
                "validation_return": validation_performance["total_return"],
                "performance_difference": round(
                    train_performance["total_return"] - validation_performance["total_return"], 2
                )
            }
        }


@router.post("/backtest/{file_id}")
async def run_backtest(
    file_id: str = Path(..., description="File ID of saved historical data"),
    strategy: str = Query("moving_average_crossover", description="Strategy name"),
    short_period: int = Query(10, description="Short MA period"),
    long_period: int = Query(30, description="Long MA period"),
    train_split: float = Query(0.8, description="Training data percentage (0.8 = 80%)"),
    initial_capital: float = Query(10000.0, description="Initial capital for backtest"),
    output_dir: str = Query("historical_data", description="Directory with JSON files")
):
    """
    Run a backtest on historical data from a saved JSON file.
    Splits data into training (80%) and validation (20%) periods.
    """
    def execute_backtest():
        # Load the historical data file
        if not os.path.exists(output_dir):
            return {"error": "Historical data directory not found"}
        
        matching_files = [f for f in os.listdir(output_dir) if f.endswith(f"_{file_id}.json")]
        
        if not matching_files:
            return {"error": f"No file found with ID {file_id}"}
        
        filepath = os.path.join(output_dir, matching_files[0])
        
        with open(filepath, "r") as f:
            file_data = json.load(f)
        
        # Extract the data array
        data = file_data.get("data", [])
        
        if not data:
            return {"error": "No data found in file"}
        
        # Initialize backtest engine
        engine = BacktestEngine(data, train_split=train_split)
        
        # Run the backtest
        results = engine.run_backtest(
            strategy_name=strategy,
            short_period=short_period,
            long_period=long_period,
            initial_capital=initial_capital
        )
        
        # Add file metadata
        results["file_info"] = {
            "file_id": file_id,
            "symbol": file_data.get("symbol"),
            "duration": file_data.get("duration"),
            "bar_size": file_data.get("bar_size")
        }
        
        return results
    
    result = await run_in_threadpool(execute_backtest)
    return JSONResponse(result)


@router.get("/optimize_strategy/{file_id}")
async def optimize_strategy(
    file_id: str = Path(..., description="File ID of saved historical data"),
    strategy: str = Query("moving_average_crossover", description="Strategy name"),
    train_split: float = Query(0.8, description="Training data percentage"),
    initial_capital: float = Query(10000.0, description="Initial capital"),
    output_dir: str = Query("historical_data", description="Directory with JSON files")
):
    """
    Optimize strategy parameters by testing multiple combinations on training data only.
    Returns the best parameters based on training performance.
    """
    def find_optimal_params():
        # Load the historical data file
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
        
        # Initialize backtest engine
        engine = BacktestEngine(data, train_split=train_split)
        
        # Test different parameter combinations
        best_params = None
        best_return = float('-inf')
        results = []
        
        # Grid search over MA periods
        for short in [5, 10, 15, 20]:
            for long in [20, 30, 50, 100]:
                if short >= long:
                    continue
                
                trades = engine.moving_average_crossover_strategy(
                    engine.train_data, short, long
                )
                performance = engine.calculate_performance(trades, initial_capital)
                
                result = {
                    "short_period": short,
                    "long_period": long,
                    "return": performance["total_return"],
                    "win_rate": performance["win_rate"],
                    "total_trades": performance["total_trades"]
                }
                results.append(result)
                
                if performance["total_return"] > best_return:
                    best_return = performance["total_return"]
                    best_params = {
                        "short_period": short,
                        "long_period": long,
                        "performance": performance
                    }
        
        return {
            "file_id": file_id,
            "symbol": file_data.get("symbol"),
            "best_parameters": best_params,
            "all_results": sorted(results, key=lambda x: x["return"], reverse=True),
            "data_summary": engine.get_data_summary()
        }
    
    result = await run_in_threadpool(find_optimal_params)
    return JSONResponse(result)
