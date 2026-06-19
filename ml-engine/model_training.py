import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import joblib
import config

def train_and_evaluate():
    print("Loading Aggregated Data for Model Training...")
    try:
        df = pd.read_csv(config.AGGREGATED_DATA_PATH)
    except FileNotFoundError:
        print("Aggregated data not found. Run data_pipeline.py first.")
        return
        
    features = [
        'Hour', 'DayOfWeek', 'Month', 'Weekend_Flag', 'Cluster_ID',
        'Vehicle_Severity', 'Violation_Severity', 'junction_encoded', 'police_encoded'
    ]
    target = 'Violation_Count'
    
    # Ensure all features exist
    available_features = [f for f in features if f in df.columns]
    
    X = df[available_features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_STATE)
    
    print(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples.")
    
    # Model 1: XGBoost
    print("\nTraining XGBoost Regressor...")
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=config.RANDOM_STATE
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_r2 = r2_score(y_test, xgb_preds)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    print(f"XGBoost R^2: {xgb_r2:.4f} | RMSE: {xgb_rmse:.4f}")
    
    # Model 2: LightGBM
    print("\nTraining LightGBM Regressor...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=config.RANDOM_STATE
    )
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict(X_test)
    lgb_r2 = r2_score(y_test, lgb_preds)
    lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_preds))
    print(f"LightGBM R^2: {lgb_r2:.4f} | RMSE: {lgb_rmse:.4f}")
    
    # Select Best Model
    best_model = None
    best_name = ""
    
    if xgb_r2 > lgb_r2:
        best_model = xgb_model
        best_name = "XGBoost"
        best_r2 = xgb_r2
        best_rmse = xgb_rmse
    else:
        best_model = lgb_model
        best_name = "LightGBM"
        best_r2 = lgb_r2
        best_rmse = lgb_rmse
        
    print(f"\nWinner: {best_name}")
    
    # Save the model
    print(f"Saving best model to {config.MODEL_SAVE_PATH}...")
    joblib.dump({
        'model': best_model,
        'model_name': best_name,
        'r2': best_r2,
        'rmse': best_rmse,
        'features': available_features
    }, config.MODEL_SAVE_PATH)
    
    print("Model Training Complete!")

if __name__ == "__main__":
    train_and_evaluate()
