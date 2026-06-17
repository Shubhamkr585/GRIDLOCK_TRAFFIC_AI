import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
from sklearn.model_selection import train_test_split
from config import config

def train_model(input_csv='enriched_training_data.csv'):
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found. Please run the full pipeline (data_ingestion -> api_osm -> api_weather -> feature_engineering) first.")
        return
        
    print(f"Loading enriched dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Feature Selection
    # Drop geohash if there are too many bins, or convert to categorical
    df['geohash'] = df['geohash'].astype('category')
    
    if 'RoadType' in df.columns:
        df['RoadType'] = df['RoadType'].astype('category')
    if 'Weather' in df.columns:
        df['Weather'] = df['Weather'].astype('category')
        
    features = [col for col in ['geohash', 'day_of_week', 'hour', 'NumberofLanes', 'RoadType', 'Temperature', 'Weather'] if col in df.columns]
    target = 'severity_score'
    
    print(f"Total training points: {len(df)}")
    print(f"Features used: {features}")
    
    X = df[features]
    y = df[target]
    
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
