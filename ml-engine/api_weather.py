import pandas as pd
import requests
import pygeohash as pgh
import time
import os
from config import config

def get_weather_data(lat, lon, date_str):
    """
    Fetches Weather Conditions using Open-Meteo API.
    """
    try:
        url = f"{config.OPEN_METEO_URL}?latitude={lat}&longitude={lon}&start_date={date_str}&end_date={date_str}&daily=temperature_2m_max,precipitation_sum&timezone=auto"
        response = requests.get(url)
        data = response.json()
        
        temp = 25.0 # default
        weather = 'Clear'
        
        if 'daily' in data:
            if 'temperature_2m_max' in data['daily'] and len(data['daily']['temperature_2m_max']) > 0:
                temp_val = data['daily']['temperature_2m_max'][0]
                if temp_val is not None:
                    temp = float(temp_val)
                    
            if 'precipitation_sum' in data['daily'] and len(data['daily']['precipitation_sum']) > 0:
                precip = data['daily']['precipitation_sum'][0]
                if precip is not None and precip > 2.0:
                    weather = 'Rainy'
                
        return {'Temperature': temp, 'Weather': weather}
    except Exception as e:
        print(f"Open-Meteo API Error for {lat},{lon} on {date_str}: {e}")
        return {'Temperature': 25.0, 'Weather': 'Clear'}

def fetch_weather_for_hotspots(input_csv='aggregated_demand.csv', output_csv='weather_features.csv'):
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return
        
    df = pd.read_csv(input_csv)
    
    # We need unique Geohash + Date combinations
    unique_combos = df[['geohash', 'date']].drop_duplicates()
    print(f"Fetching Weather data for {len(unique_combos)} unique Hotspot/Date combinations...")
    
    weather_records = []
    
    for i, row in unique_combos.iterrows():
        ghash = row['geohash']
        date_str = row['date']
        lat, lon = pgh.decode(ghash)
        
        weather_data = get_weather_data(float(lat), float(lon), str(date_str))
        weather_data['geohash'] = ghash
        weather_data['date'] = str(date_str)
        weather_records.append(weather_data)
        
        # Free API limit: 10,000 per day. Small sleep to avoid bursts.
        time.sleep(0.1)
        if (i+1) % 50 == 0:
            print(f"Processed {i+1} / {len(unique_combos)} weather records...")
            
    weather_df = pd.DataFrame(weather_records)
    weather_df.to_csv(output_csv, index=False)
    print(f"Weather feature extraction complete. Saved to {output_csv}")

if __name__ == "__main__":
    fetch_weather_for_hotspots()
