import asyncio
import os
from fastapi import Path, Query
from ib_insync import IB, Stock, MarketOrder
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt
from datetime import datetime
from mock_data import MockIBKRService
from services.IKBRClient import IBKRClient
from historical_json import router as historical_json_router
from backtest import router as backtest_router


IB_PORT = 4002  # 4001 for live trading, 4002 for paper trading
IB_HOST = '127.0.0.1'
USE_MOCK = os.getenv('USE_MOCK', 'false').lower() == 'true'
app = FastAPI()
# Set rate_limit_delay to 0.1 seconds (10 requests/sec for CP Gateway, or 0.02 for 50 req/sec)
ib_client = IBKRClient(host=IB_HOST, port=IB_PORT, client_id=1, rate_limit_delay=0.1)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:4200"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(historical_json_router, prefix="/json", tags=["Historical JSON"])
app.include_router(backtest_router, prefix="/backtest", tags=["Backtesting"])

def get_ibkr_data():
    price = ib_client.get_realtime_price('MSFT')
    return {"connected": True, "symbol": "MSFT", "latest_price": price}



@app.get("/historical_stock/{symbol}")
async def historical_stock(symbol: str = Path(...)):
    def fetch():
        bars = ib_client.get_historical_data(
            symbol=symbol,
            duration='1 M',
            bar_size='1 day',
            what_to_show='TRADES',
        )
        return [
            {
                "date": bar.date.strftime('%Y-%m-%d'),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]

    return JSONResponse(await run_in_threadpool(fetch))


def execute_swing_trade():
    price = ib_client.get_realtime_price('AAPL')
    order = MarketOrder('SELL' if price > 150 else 'BUY', 10)
    trade = ib_client.place_market_order(Stock('AAPL', 'SMART', 'USD'), order)
    return {
        "symbol": 'AAPL',
        "action": order.action,
        "quantity": order.totalQuantity,
        "price": price,
        "status": trade.orderStatus.status,
    }

def get_account_balance():
    summary = ib_client.get_account_summary()
    balance = next((item.value for item in summary if item.tag == 'NetLiquidation'), None)
    return {"balance": balance}

@app.get("/earliest_data/{symbol}")
async def get_earliest_data(symbol: str = Path(...)):
    if USE_MOCK:
        return JSONResponse(MockIBKRService.get_mock_earliest_data(symbol))

    def fetch():
        timestamp = ib_client.get_head_timestamp(symbol, what_to_show='TRADES')
        return {
            "symbol": symbol.upper(),
            "earliest_data_date": timestamp.strftime('%Y-%m-%d') if timestamp else None,
        }

    return JSONResponse(await run_in_threadpool(fetch))


@app.get("/historical_stock_graph/{symbol}")
async def historical_stock_graph(
    symbol: str = Path(..., description="Stock symbol"),
    duration: str = Query("1 M", description="Duration string (e.g., '1 M', '1 Y')"),
    bar_size: str = Query("1 day", description="Bar size string (e.g., '1 day', '1 hour')")
):
    """
    Generates and returns a PNG graph of historical stock data.
    """
    def generate_and_get_path():
        # This synchronous function calls the method on the client
        return ib_client.generate_historical_graph(
            symbol=symbol,
            duration=duration,
            bar_size=bar_size
        )

    # Run the blocking I/O operation in a thread pool
    file_path = await run_in_threadpool(generate_and_get_path)

    if not file_path or not os.path.exists(file_path):
        return JSONResponse(
            status_code=404,
            content={"message": f"Could not generate graph for {symbol}. No data found."}
        )

    return FileResponse(file_path, media_type="image/png")



@app.get("/swing_simulation/{symbol}")
async def swing_simulation(
    symbol: str = Path(..., description="Stock symbol"),
    duration: str = Query("1 Y", description="Duration string (e.g., '1 Y', '6 M')"),
    bar_size: str = Query("1 day", description="Bar size string (e.g., '1 day')"),
    short_ma: int = Query(10, description="Short moving average period"),
    long_ma: int = Query(30, description="Long moving average period")
):
    """
    Generates and returns a PNG graph of swing trade simulation.
    Uses only first 50% of historical data for backtesting.
    """
    def generate_simulation():
        return ib_client.simulate_swing_trade(
            symbol=symbol,
            duration=duration,
            bar_size=bar_size,
            short_ma=short_ma,
            long_ma=long_ma
        )

    # Run the simulation in a thread pool (since it involves I/O and calculations)
    file_path = await run_in_threadpool(generate_simulation)

    if not file_path or not os.path.exists(file_path):
        return JSONResponse(
            status_code=404,
            content={"message": f"Could not generate swing simulation for {symbol}. No data found."}
        )

    return FileResponse(file_path, media_type="image/png")



@app.get("/add")
async def add_numbers(a: float = Query(...), b: float = Query(...)):
    """
    Test endpoint that adds two numbers.
    Example: /add?a=5&b=7 returns {"result": 12}
    """
    return {"result": a + b}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)