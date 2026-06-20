import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import config
import warnings
import sys
import io

# Force UTF-8 encoding for Windows console to support emojis
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

def train_and_evaluate():
    print("Phase 7: Model Training & Rigorous Evaluation...")
    try:
        df = pd.read_csv(config.AGGREGATED_DATA_PATH)
    except FileNotFoundError:
        print("Aggregated data not found. Run data_pipeline.py first.")
        return

    features = [
        'cluster_centroid_lat', 'cluster_centroid_lon', 
        'hour_sin', 'hour_cos', 
        'violations_lag_1h', 'rolling_3h_mean', 
        'avg_vehicle_weight', 'antigravity_repulsion_factor'
    ]
    target = 'Target_Severity'
    
    # Chronological Split
    df['Date_Hour'] = pd.to_datetime(df['Date_Hour'])
    df = df.sort_values('Date_Hour')
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Model 1: LightGBM
    lgb_model = lgb.LGBMRegressor(random_state=42, verbosity=-1)
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict(X_test)
    lgb_r2 = r2_score(y_test, lgb_preds)
    lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_preds))
    lgb_mae = mean_absolute_error(y_test, lgb_preds)
    
    # Model 2: XGBoost
    xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_r2 = r2_score(y_test, xgb_preds)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    
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
    
    # Save the winning model and metrics so Streamlit can load them!
    import joblib
    best_model = lgb_model if winner == "LightGBM" else xgb_model
    best_r2 = lgb_r2 if winner == "LightGBM" else xgb_r2
    best_rmse = lgb_rmse if winner == "LightGBM" else xgb_rmse
    
    joblib.dump({
        'model': best_model,
        'features': features,
        'model_name': winner,
        'r2': best_r2,
        'rmse': best_rmse
    }, config.MODEL_SAVE_PATH)
    print("Model successfully saved for Streamlit!")
    
if __name__ == "__main__":
    train_and_evaluate()
