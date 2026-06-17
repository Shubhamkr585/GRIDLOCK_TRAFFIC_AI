import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import pygeohash as pgh
import os
from sklearn.model_selection import train_test_split
from config import config

def calculate_cim(row):
    """
    Calculate Congestion-Impact Multiplier based on demand.
    Since we don't have RoadType yet, we use a basic proxy logic for severity.
    """
    base_demand = row['demand']
    
    # Placeholder for OSM API logic (Aman will implement)
    # For now, severity score is directly proportional to demand
    return base_demand * 1.2

def process_raw_dataset(file_path):
    print(f"Loading 104MB raw violation data from {file_path}...")
    
    # Use chunking if memory is an issue, but 104MB should easily fit in RAM
    df = pd.read_csv(file_path, usecols=['latitude', 'longitude', 'created_datetime'])
    
    print("Dropping missing coordinates...")
    df = df.dropna(subset=['latitude', 'longitude', 'created_datetime'])
    
    print("Computing Geohashes (Precision 6 = ~1.2km x 600m)...")
    df['geohash'] = df.apply(lambda row: pgh.encode(row['latitude'], row['longitude'], precision=6), axis=1)
    
    print("Extracting Temporal Features...")
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce')
    df = df.dropna(subset=['created_datetime'])
    df['day_of_week'] = df['created_datetime'].dt.dayofweek
    df['hour'] = df['created_datetime'].dt.hour
    
    print("Aggregating into Demand (Hotspots)...")
    # Group by location and time to get 'demand' (count of violations)
    aggregated_df = df.groupby(['geohash', 'day_of_week', 'hour']).size().reset_index(name='demand')
    
    return aggregated_df

def load_and_prepare_data(file_path):
    df = process_raw_dataset(file_path)
    
    print("Calculating Congestion Impact Multiplier (Severity Score)...")
    df['severity_score'] = df.apply(calculate_cim, axis=1)
    
    # Convert categorical to category type for LightGBM
    df['geohash'] = df['geohash'].astype('category')
        
    features = ['geohash', 'day_of_week', 'hour']
    target = 'severity_score'
    
    return df[features], df[target]

def train_model():
    X, y = load_and_prepare_data(config.DATASET_PATH)
    
    print(f"Total aggregated Hotspot data points: {len(X)}")
    print("Splitting dataset...")
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
    print("Training LightGBM model...")
    model = lgb.train(
        config.LGBM_PARAMS,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
    )
    
    print("Saving model to lgbm_model.pkl...")
    joblib.dump(model, 'lgbm_model.pkl')
    print("Training complete! Model is ready for FastAPI inference.")

if __name__ == "__main__":
    train_model()
