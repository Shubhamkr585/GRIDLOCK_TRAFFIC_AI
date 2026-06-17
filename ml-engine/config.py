import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Dataset path (assuming it sits in the project root)
    DATASET_PATH = os.getenv("DATASET_PATH", "../dataset/train.csv")
    
    # OSM Overpass API URL
    OVERPASS_URL = "http://overpass-api.de/api/interpreter"
    
    # Open-Meteo API URL
    OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    # LightGBM Params
    LGBM_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.8
    }

config = Config()
