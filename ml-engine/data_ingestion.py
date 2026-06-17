import pandas as pd
import pygeohash as pgh
import os
from config import config

def ingest_and_aggregate(file_path):
    print(f"Loading raw dataset from {file_path}...")
    df = pd.read_csv(file_path, usecols=['latitude', 'longitude', 'created_datetime'])
    
    print("Dropping invalid data...")
    df = df.dropna(subset=['latitude', 'longitude', 'created_datetime'])
    
    print("Computing Geohashes (Precision 6)...")
    df['geohash'] = df.apply(lambda row: pgh.encode(row['latitude'], row['longitude'], precision=6), axis=1)
    
    print("Extracting Time Features...")
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce')
    df = df.dropna(subset=['created_datetime'])
    df['date'] = df['created_datetime'].dt.date
    df['day_of_week'] = df['created_datetime'].dt.dayofweek
    df['hour'] = df['created_datetime'].dt.hour
    
    print("Aggregating Traffic Volume (Demand)...")
    # We keep date to map historical weather later, but group by day_of_week/hour for modeling
    aggregated = df.groupby(['geohash', 'date', 'day_of_week', 'hour']).size().reset_index(name='demand')
    
    output_path = 'aggregated_demand.csv'
    aggregated.to_csv(output_path, index=False)
    print(f"Saved aggregated dataset to {output_path}")
    
    return aggregated

if __name__ == "__main__":
    ingest_and_aggregate(config.DATASET_PATH)
