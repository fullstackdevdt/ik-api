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

@router.get("/analyze_file/{file_id}")
async def analyze_file(
    file_id: str = Path(..., description="The ID of the saved file to analyze"),
    output_dir: str = Query("historical_data", description="Directory where files are")
):
    """
    Demonstrates how to process the 'data' array in Python.
    Calculates average price, total volume, and identifies days where price closed higher than open.
    """
    def process_analysis():
        # --- PART 1: Find and Load the File ---
        if not os.path.exists(output_dir):
            return {"error": "Directory not found"}
            
        # List comprehension to find the specific file by ID
        target_files = [f for f in os.listdir(output_dir) if f.endswith(f"_{file_id}.json")]
        
        if not target_files:
            return {"error": "File ID not found"}
            
        filepath = os.path.join(output_dir, target_files[0])
        
        with open(filepath, "r") as f:
            full_json = json.load(f)
            
        # Get the list (array) of data points
        # If "data" key is missing, return an empty list []
        data_list = full_json.get("data", [])

        # --- PART 2: Python Data Processing ---
        
        # Initialize variables to hold our calculations
        total_close_price = 0
        total_volume = 0
        highest_volume_seen = 0
        highest_volume_date = ""
        green_days_count = 0  # Days where Close > Open

        # We can create a new list to store simplified data
        processed_days = []

        # Iterate (loop) through each item in the list
        # 'day_data' represents one dictionary inside the array
        for day_data in data_list:
            
            # 1. Extract values using dictionary keys
            close_price = day_data["close"]
            open_price = day_data["open"]
            volume = day_data["volume"]
            date = day_data["date"]

            # 2. Accumulate totals (for averages later)
            total_close_price = total_close_price + close_price
            total_volume = total_volume + volume

            # 3. Check for specific conditions (Logic)
            # Was this the highest volume day so far?
            if volume > highest_volume_seen:
                highest_volume_seen = volume
                highest_volume_date = date

            # Was it a "Green Day"? (price went up during the day)
            is_green_day = close_price > open_price
            if is_green_day:
                green_days_count += 1
            
            # 4. Create a custom structure/transformation
            # Let's say we want to calculate the 'spread' (High - Low)
            spread = day_data["high"] - day_data["low"]
            
            # Store this processed info
            processed_days.append({
                "date": date,
                "is_green": is_green_day,
                "price_spread": round(spread, 2), # Round to 2 decimals
                "close": close_price
            })

        # --- PART 3: Final aggregate calculations ---
        
        number_of_days = len(data_list)
        
        # Avoid division by zero if list is empty
        if number_of_days > 0:
            average_close = total_close_price / number_of_days
            average_volume = total_volume / number_of_days
        else:
            average_close = 0
            average_volume = 0

        # Construct the final response
        return {
            "analysis_target": full_json.get("symbol"),
            "total_days_analyzed": number_of_days,
            "averages": {
                "average_close_price": round(average_close, 2),
                "average_daily_volume": int(average_volume)
            },
            "highlights": {
                "highest_volume_date": highest_volume_date,
                "highest_volume": highest_volume_seen,
                "green_days_count": green_days_count,
                "red_days_count": number_of_days - green_days_count
            },
            # Return our custom processed list (showing first 5 items to keep JSON small)
            "processed_data_preview": processed_days[:5]
        }

    return await run_in_threadpool(process_analysis)


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