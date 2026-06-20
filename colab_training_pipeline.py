# ==============================================================================
# Flipkart Gridlock (Theme 1) | Target: Congestion Severity Prediction
# ==============================================================================
# Google Colab Training Pipeline
# 1. Upload your 'violations.csv' to the Colab environment.
# 2. Run the following command in a notebook cell to install dependencies:
#    !pip install pandas numpy scikit-learn xgboost lightgbm
# 3. Paste this entire script into a cell and execute it.
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

RAW_DATA_PATH = 'violations.csv'

def load_and_clean_data(filepath):
    print("Phase 1: Data Ingestion & Memory Safeguards...")
    df = pd.read_csv(filepath)
    
    if 'latitude' in df.columns: df['latitude'] = df['latitude'].astype('float32')
    if 'longitude' in df.columns: df['longitude'] = df['longitude'].astype('float32')
        
    if 'vehicle_type' in df.columns: df['vehicle_type'] = df['vehicle_type'].astype('category')
    if 'police_station' in df.columns: df['police_station'] = df['police_station'].astype('category')
        
    print("Phase 2: Data Cleaning & Anti-Leakage Filtering...")
    df = df.dropna(subset=['latitude', 'longitude'])
    df = df[(df['latitude'] != 0.0) & (df['longitude'] != 0.0)]
    
    if 'validation_status' in df.columns:
        df = df[df['validation_status'].astype(str).str.lower() == 'approved']
        
    leakage_cols = ['id', 'closed_date', 'updated_timestamp', 'validator_id', 'data_sent_to_scita_timestamp']
    df = df.drop(columns=[col for col in leakage_cols if col in df.columns])
    
    cat_cols = ['offence_code', 'violation_type', 'vehicle_type', 'police_station', 'junction_name']
    for col in cat_cols:
        if col in df.columns:
            if df[col].dtype.name == 'category':
                df[col] = df[col].cat.add_categories('UNKNOWN').fillna('UNKNOWN')
            else:
                df[col] = df[col].fillna('UNKNOWN')
    
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce', utc=True)
    df = df.dropna(subset=['created_datetime'])
    df = df.sort_values('created_datetime').reset_index(drop=True)
    return df

def spatial_clustering(df):
    print("Phase 3: Spatial Intelligence & Advanced DBSCAN...")
    coords = np.radians(df[['latitude', 'longitude']].values)
    eps_rad = 0.05 / 6371.0088
    
    db = DBSCAN(eps=eps_rad, min_samples=10, metric='haversine', algorithm='ball_tree').fit(coords)
    df['Cluster_ID'] = db.labels_
    df = df[df['Cluster_ID'] != -1]
    
    centroids = df.groupby('Cluster_ID')[['latitude', 'longitude']].mean().rename(
        columns={'latitude': 'cluster_centroid_lat', 'longitude': 'cluster_centroid_lon'}
    )
    df = df.merge(centroids, on='Cluster_ID', how='left')
    return df

def time_series_transformation(df):
    print("Phase 4, 5 & 6: Time-Series Engine, Antigravity & Target Synthesis...")
    df['Date_Hour'] = df['created_datetime'].dt.floor('h')
    
    def get_vehicle_weight(v):
        v = str(v).lower().strip()
        if v in ['scooter', 'motor cycle', 'moped', 'two wheeler', 'auto rickshaw', 'passenger auto', 'goods auto']: return 1
        elif v in ['car', 'jeep', 'van', 'suv', 'maxi-cab']: return 3
        else: return 6
    
    df['vehicle_weight'] = df['vehicle_type'].apply(get_vehicle_weight).astype('float32')
    
    block_agg = df.groupby(['Cluster_ID', 'Date_Hour']).agg({
        'created_datetime': 'count', 
        'vehicle_weight': 'mean',
        'cluster_centroid_lat': 'first',
        'cluster_centroid_lon': 'first'
    }).rename(columns={'created_datetime': 'Violation_Count', 'vehicle_weight': 'avg_vehicle_weight'}).reset_index()
    
    min_date, max_date = block_agg['Date_Hour'].min(), block_agg['Date_Hour'].max()
    all_hours = pd.date_range(min_date, max_date, freq='h')
    clusters = block_agg['Cluster_ID'].unique()
    
    block_agg = block_agg.set_index(['Cluster_ID', 'Date_Hour'])
    multi_idx = pd.MultiIndex.from_product([clusters, all_hours], names=['Cluster_ID', 'Date_Hour'])
    dense_df = block_agg.reindex(multi_idx).reset_index()
    
    dense_df['Violation_Count'] = dense_df['Violation_Count'].fillna(0)
    dense_df[['cluster_centroid_lat', 'cluster_centroid_lon']] = dense_df.groupby('Cluster_ID')[['cluster_centroid_lat', 'cluster_centroid_lon']].ffill().bfill()
    
    dense_df['hour_of_day'] = dense_df['Date_Hour'].dt.hour
    dense_df['hour_sin'] = np.sin(2 * np.pi * dense_df['hour_of_day'] / 24)
    dense_df['hour_cos'] = np.cos(2 * np.pi * dense_df['hour_of_day'] / 24)
    
    dense_df['avg_vehicle_weight'] = dense_df['avg_vehicle_weight'].fillna(1.0)
    
    dense_df['violations_lag_1h'] = dense_df.groupby('Cluster_ID')['Violation_Count'].shift(1).fillna(0)
    dense_df['rolling_3h_mean'] = dense_df.groupby('Cluster_ID')['violations_lag_1h'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    
    cluster_density = df.groupby('Cluster_ID').size() / (np.pi * (0.05)**2)
    dense_df['cluster_density'] = dense_df['Cluster_ID'].map(cluster_density)
    
    dense_df['antigravity_repulsion_factor'] = np.exp(-1 / (dense_df['violations_lag_1h'] + 0.1)) * dense_df['cluster_density']
    
    dense_df['Raw_Score'] = (dense_df['Violation_Count'] * dense_df['avg_vehicle_weight']) + (dense_df['rolling_3h_mean'] * 1.5)
    pct_90 = dense_df['Raw_Score'].quantile(0.90)
    if pct_90 == 0: pct_90 = 1
    dense_df['Target_Severity'] = 100 * (1 - np.exp(-dense_df['Raw_Score'] / pct_90))
    
    return dense_df

def train_and_evaluate(df):
    print("Phase 7: Model Training & Rigorous Evaluation...")
    features = [
        'cluster_centroid_lat', 'cluster_centroid_lon', 
        'hour_sin', 'hour_cos', 
        'violations_lag_1h', 'rolling_3h_mean', 
        'avg_vehicle_weight', 'antigravity_repulsion_factor'
    ]
    target = 'Target_Severity'
    
    df['Date_Hour'] = pd.to_datetime(df['Date_Hour'])
    df = df.sort_values('Date_Hour')
    df = df.dropna(subset=['cluster_centroid_lat'])
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    lgb_model = lgb.LGBMRegressor(random_state=42, verbosity=-1)
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict(X_test)
    lgb_r2, lgb_rmse, lgb_mae = r2_score(y_test, lgb_preds), np.sqrt(mean_squared_error(y_test, lgb_preds)), mean_absolute_error(y_test, lgb_preds)
    
    xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_r2, xgb_rmse, xgb_mae = r2_score(y_test, xgb_preds), np.sqrt(mean_squared_error(y_test, xgb_preds)), mean_absolute_error(y_test, xgb_preds)
    
    winner = "LightGBM" if lgb_r2 > xgb_r2 else "XGBoost"
    
    print("\n=========================================================")
    print("🏆 FLIPKART GRIDLOCK: MODEL PERFORMANCE EVALUATION 🏆")
    print("=========================================================")
    print("Target Metric: Congestion Severity (0-100)")
    print("Evaluation Set: Chronological Unseen Future Data (20%)\n")
    
    print("▶ MODEL 1: LightGBM")
    print(f"  - R² Score : {lgb_r2:.4f}")
    print(f"  - RMSE     : {lgb_rmse:.4f}")
    print(f"  - MAE      : {lgb_mae:.4f}\n")
    
    print("▶ MODEL 2: XGBoost")
    print(f"  - R² Score : {xgb_r2:.4f}")
    print(f"  - RMSE     : {xgb_rmse:.4f}")
    print(f"  - MAE      : {xgb_mae:.4f}\n")
    
    print(f"✅ WINNER: [{winner}] selected for Streamlit Deployment.")
    print("=========================================================\n")

if __name__ == "__main__":
    df = load_and_clean_data(RAW_DATA_PATH)
    if not df.empty:
        df = spatial_clustering(df)
        final_df = time_series_transformation(df)
        train_and_evaluate(final_df)
