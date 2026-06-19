import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
import config

def load_and_clean_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print("Dataset not found. Please ensure the CSV exists.")
        return pd.DataFrame()

    # Data Cleaning Rules
    initial_rows = len(df)
    
    # 1. Drop missing lat/long
    df = df.dropna(subset=['latitude', 'longitude'])
    
    # 2. Convert and drop missing timestamps
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce', utc=True)
    df = df.dropna(subset=['created_datetime'])
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # 3. Retain only 'approved' validation_status (case-insensitive)
    if 'validation_status' in df.columns:
        df = df[df['validation_status'].astype(str).str.lower() == 'approved']
        
    print(f"Cleaned data: {len(df)} rows retained from original {initial_rows} rows.")
    return df

def temporal_engineering(df):
    print("Extracting Temporal Features...")
    df['Hour'] = df['created_datetime'].dt.hour
    df['DayOfWeek'] = df['created_datetime'].dt.dayofweek
    df['Month'] = df['created_datetime'].dt.month
    df['Weekend_Flag'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    return df

def spatial_clustering(df):
    print("Running DBSCAN Spatial Clustering...")
    coords = df[['latitude', 'longitude']].values
    # eps = 0.0005 (approx 50m radius)
    db = DBSCAN(eps=config.DBSCAN_EPS, min_samples=config.DBSCAN_MIN_SAMPLES, metric='euclidean').fit(coords)
    df['Cluster_ID'] = db.labels_
    
    # Drop noise points (-1)
    df = df[df['Cluster_ID'] != -1]
    
    print(f"Discovered {df['Cluster_ID'].nunique()} valid spatial clusters.")
    
    # Save Cluster Centroids for Dashboard
    centroids = df.groupby('Cluster_ID')[['latitude', 'longitude']].mean().reset_index()
    centroids.to_csv(config.CLUSTER_GEO_PATH, index=False)
    
    return df

def severity_encoding(df):
    print("Applying Severity Encodings...")
    
    def get_vehicle_weight(v):
        if pd.isna(v): return config.DEFAULT_VEHICLE_WEIGHT
        v = str(v).lower()
        return config.VEHICLE_WEIGHTS.get(v, config.DEFAULT_VEHICLE_WEIGHT)
        
    def get_violation_weight(v):
        if pd.isna(v): return config.DEFAULT_VIOLATION_WEIGHT
        v = str(v).lower()
        return config.VIOLATION_WEIGHTS.get(v, config.DEFAULT_VIOLATION_WEIGHT)
        
    df['Vehicle_Severity'] = df['vehicle_type'].apply(get_vehicle_weight)
    df['Violation_Severity'] = df['violation_type'].apply(get_violation_weight)
    
    return df

def aggregate_features(df):
    print("Aggregating Features for Model Training...")
    
    # Handle missing categoricals
    df['junction_name'] = df['junction_name'].fillna('Unknown')
    df['police_station'] = df['police_station'].fillna('Unknown')
    
    # Group by spatial-temporal keys
    group_cols = ['Cluster_ID', 'Hour', 'DayOfWeek', 'Month', 'Weekend_Flag']
    
    # Calculate aggregations
    agg_funcs = {
        'id': 'count', # This counts violations
        'Vehicle_Severity': 'mean',
        'Violation_Severity': 'mean',
        'junction_name': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
        'police_station': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
    }
    
    agg_df = df.groupby(group_cols).agg(agg_funcs).reset_index()
    agg_df = agg_df.rename(columns={'id': 'Violation_Count'})
    
    # Encode junction_name and police_station
    le_junction = LabelEncoder()
    le_police = LabelEncoder()
    agg_df['junction_encoded'] = le_junction.fit_transform(agg_df['junction_name'])
    agg_df['police_encoded'] = le_police.fit_transform(agg_df['police_station'])
    
    # Drop raw text
    agg_df = agg_df.drop(columns=['junction_name', 'police_station'])
    
    return agg_df

def run_pipeline():
    df = load_and_clean_data(config.RAW_DATA_PATH)
    if df.empty:
        return
        
    df = temporal_engineering(df)
    df = spatial_clustering(df)
    df = severity_encoding(df)
    
    # Save the processed but un-aggregated data for the Impact Engine
    df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    
    # Create Aggregation for Model Training
    agg_df = aggregate_features(df)
    agg_df.to_csv(config.AGGREGATED_DATA_PATH, index=False)
    
    print("Data Pipeline Execution Complete!")

if __name__ == "__main__":
    run_pipeline()
