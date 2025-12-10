from fastapi import APIRouter, Path, Query
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
import json
import os
import time

router = APIRouter()

@router.post("/save_historical/{symbol}")
async def save_historical(
    symbol: str = Path(..., description="Stock symbol"),
    duration: str = Query("1 M", description="Duration string (e.g., '1 M', '1 Y', '2 Y')"),
    bar_size: str = Query("1 day", description="Bar size string (e.g., '1 day', '1 hour', '5 mins')"),
    output_dir: str = Query("historical_data", description="Directory to save JSON files")
):
    """
    Fetches historical data from IBKR and saves it to a JSON file with a timestamp ID.
    """
    from main import ib_client  # Import the shared client
    
    def get_and_save():
        # Fetch historical data
        bars = ib_client.get_historical_data(
            symbol=symbol,
            duration=duration,
            bar_size=bar_size,
            what_to_show='TRADES',
        )
        
        if not bars:
            return {"error": f"No data found for {symbol}"}
        
        # Generate unique ID (current time in milliseconds)
        file_id = str(int(time.time() * 1000))
        
        # Convert to JSON-serializable format
        data = [
            {
                "date": bar.date.strftime('%Y-%m-%d %H:%M:%S'),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate filename with ID
        filename = f"{symbol.lower()}_{file_id}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Create metadata object to save
        save_data = {
            "id": file_id,
            "symbol": symbol.upper(),
            "duration": duration,
            "bar_size": bar_size,
            "data_points": len(data),
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "data": data
        }
        
        # Save to JSON
        with open(filepath, "w") as f:
            json.dump(save_data, f, indent=2)
        
        return {
            "message": "Data saved successfully",
            "id": file_id,
            "symbol": symbol.upper(),
            "duration": duration,
            "bar_size": bar_size,
            "data_points": len(data),
            "filepath": filepath
        }
    
    result = await run_in_threadpool(get_and_save)
    return JSONResponse(result)


@router.get("/load_historical/{file_id}")
async def load_historical(
    file_id: str = Path(..., description="File ID (timestamp) from save_historical"),
    output_dir: str = Query("historical_data", description="Directory where JSON files are stored")
):
    """
    Loads historical data from a JSON file using its ID.
    """
    def load_data():
        # Find file by searching for the ID in the directory
        if not os.path.exists(output_dir):
            return {"error": "Historical data directory not found"}
        
        # Look for file matching the pattern *_{file_id}.json
        matching_files = [f for f in os.listdir(output_dir) if f.endswith(f"_{file_id}.json")]
        
        if not matching_files:
            return {
                "error": "File not found",
                "message": f"No saved data found with ID {file_id}"
            }
        
        filepath = os.path.join(output_dir, matching_files[0])
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        return data
    
    result = await run_in_threadpool(load_data)
    return JSONResponse(result)


@router.get("/list_saved_historical")
async def list_saved_historical(
    output_dir: str = Query("historical_data", description="Directory where JSON files are stored")
):
    """
    Lists all saved historical data JSON files with their metadata.
    """
    def list_files():
        if not os.path.exists(output_dir):
            return {"files": [], "message": "No historical data directory found"}
        
        json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
        
        # Read metadata from each file
        file_list = []
        for filename in json_files:
            filepath = os.path.join(output_dir, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    file_list.append({
                        "filename": filename,
                        "id": data.get("id"),
                        "symbol": data.get("symbol"),
                        "duration": data.get("duration"),
                        "bar_size": data.get("bar_size"),
                        "data_points": data.get("data_points"),
                        "created_at": data.get("created_at")
                    })
            except:
                # Skip files that can't be read
                pass
        
        return {
            "count": len(file_list),
            "files": file_list,
            "directory": output_dir
        }
    
    result = await run_in_threadpool(list_files)
    return JSONResponse(result)