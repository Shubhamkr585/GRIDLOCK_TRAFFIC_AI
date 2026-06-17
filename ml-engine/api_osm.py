import pandas as pd
import requests
import pygeohash as pgh
import time
import os
from config import config

def get_osm_data(lat, lon):
    """
    Fetches Road Width and Lane Count using OSM Overpass API.
    """
    # Define a bounding box around the coordinate (roughly 50 meters)
    bbox = f"{lat-0.0005},{lon-0.0005},{lat+0.0005},{lon+0.0005}"
    query = f"""
    [out:json];
    way({bbox})["highway"];
    out tags;
    """
    try:
        response = requests.post(config.OVERPASS_URL, data={'data': query})
        data = response.json()
        
        # Default fallbacks
        lanes = 2
        road_type = 'residential'
        
        if 'elements' in data and len(data['elements']) > 0:
            tags = data['elements'][0].get('tags', {})
            if 'lanes' in tags:
                lanes = int(tags['lanes'].split(';')[0]) # Handle lists like "2;3"
            if 'highway' in tags:
                road_type = tags['highway']
                
        return {'RoadType': road_type, 'NumberofLanes': lanes}
    except Exception as e:
        print(f"OSM API Error for {lat},{lon}: {e}")
        return {'RoadType': 'residential', 'NumberofLanes': 2}

def fetch_osm_for_hotspots(input_csv='aggregated_demand.csv', output_csv='osm_features.csv'):
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return
        
    df = pd.read_csv(input_csv)
    unique_geohashes = df['geohash'].unique()
    print(f"Fetching OSM data for {len(unique_geohashes)} unique Hotspots...")
    
    osm_records = []
    
    for i, ghash in enumerate(unique_geohashes):
        lat, lon = pgh.decode(ghash)
        osm_data = get_osm_data(float(lat), float(lon))
        osm_data['geohash'] = ghash
        osm_records.append(osm_data)
        
        # To respect public Overpass API rate limits (1 request per second)
        time.sleep(1)
        if (i+1) % 50 == 0:
            print(f"Processed {i+1} / {len(unique_geohashes)} hotspots...")
            
    osm_df = pd.DataFrame(osm_records)
    osm_df.to_csv(output_csv, index=False)
    print(f"OSM feature extraction complete. Saved to {output_csv}")

if __name__ == "__main__":
    fetch_osm_for_hotspots()
