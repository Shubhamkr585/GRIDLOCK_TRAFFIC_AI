from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import pygeohash as pgh
import os

app = FastAPI(title="GridLock AI ML Engine", description="Forecasting Traffic Congestion Severity")

# Load model globally on startup
MODEL_PATH = "lgbm_model.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model file {MODEL_PATH} not found. Please train the model first.")

class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    day_of_week: int
    hour: int

@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict_severity(req: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the model first.")
        
    # Calculate geohash from incoming coordinates
    geohash_val = pgh.encode(req.latitude, req.longitude, precision=6)
    
    # Create DataFrame matching training features
    input_data = pd.DataFrame([{
        'geohash': geohash_val,
        'day_of_week': req.day_of_week,
        'hour': req.hour
    }])
    
    # Convert categorical to match LightGBM requirement
    input_data['geohash'] = input_data['geohash'].astype('category')
    
    # Predict
    try:
        prediction = model.predict(input_data)
        severity_score = float(prediction[0])
        
        # Determine Alert Level based on score
        alert_level = "LOW"
        if severity_score > 30:
            alert_level = "CRITICAL"
        elif severity_score > 15:
            alert_level = "MODERATE"
            
        return {
            "latitude": req.latitude,
            "longitude": req.longitude,
            "geohash": geohash_val,
            "severity_score": round(severity_score, 2),
            "alert_level": alert_level
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
