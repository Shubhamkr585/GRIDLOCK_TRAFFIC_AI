# Configuration Settings for Parking Intelligence Platform

# DBSCAN Spatial Clustering Hyperparameters
# eps is in degrees (approx 50 meters in lat/long near equator)
# This controls the geographic radius of a parking hotspot
DBSCAN_EPS = 0.0005 
DBSCAN_MIN_SAMPLES = 5

# Categorical Encodings (Vehicle Severity)
VEHICLE_WEIGHTS = {
    'two wheeler': 1,
    'auto rickshaw': 2,
    'car': 3,
    'suv': 4,
    'bus': 5,
    'truck': 6,
    'tanker': 7
}

DEFAULT_VEHICLE_WEIGHT = 2

# Categorical Encodings (Violation Severity)
VIOLATION_WEIGHTS = {
    'minor violation': 1,
    'wrong parking': 3,
    'improper parking': 3,
    'footpath parking': 4,
    'footpath encroachment': 4,
    'roadside obstruction': 4,
    'blocking junction': 5,
    'no parking zone': 5
}

DEFAULT_VIOLATION_WEIGHT = 3

# File Paths
RAW_DATA_PATH = '../dataset/violations.csv'
PROCESSED_DATA_PATH = 'data_processed.csv'
AGGREGATED_DATA_PATH = 'data_aggregated.csv'
CLUSTER_GEO_PATH = 'cluster_centroids.csv'
IMPACT_SCORES_PATH = 'impact_scores.csv'
MODEL_SAVE_PATH = 'best_parking_model.pkl'

# Random State for reproducibility
RANDOM_STATE = 42
