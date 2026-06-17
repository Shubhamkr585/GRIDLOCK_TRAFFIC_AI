import pandas as pd
import os

def calculate_cim(row):
    """
    Calculate Congestion-Impact Multiplier based on demand, road type, and lanes.
    """
    base_demand = row['demand']
    
    # Road vulnerability multiplier
    road_multiplier = 1.0
    road_type = str(row.get('RoadType', 'residential')).lower()
    
    if 'residential' in road_type:
        road_multiplier *= 1.5
    elif 'primary' in road_type or 'highway' in road_type:
        road_multiplier *= 0.8
        
    # Lane vulnerability multiplier
    lanes = row.get('NumberofLanes', 2)
    if pd.isna(lanes):
        lanes = 2
    else:
        lanes = float(lanes)
        
    if lanes <= 1:
        lane_multiplier = 2.0
    elif lanes == 2:
        lane_multiplier = 1.5
    else:
        lane_multiplier = 1.0
        
    return base_demand * road_multiplier * lane_multiplier

def merge_and_engineer(base_csv='aggregated_demand.csv', osm_csv='osm_features.csv', weather_csv='weather_features.csv', output_csv='enriched_training_data.csv'):
    print("Loading base aggregated dataset...")
    df = pd.read_csv(base_csv)
    
    if os.path.exists(osm_csv):
        print("Merging OSM features...")
        osm_df = pd.read_csv(osm_csv)
        df = pd.merge(df, osm_df, on='geohash', how='left')
    else:
        print("Warning: OSM features not found. Proceeding without them.")
        
    if os.path.exists(weather_csv):
        print("Merging Weather features...")
        weather_df = pd.read_csv(weather_csv)
        # Convert date to string for safe merge
        df['date'] = df['date'].astype(str)
        weather_df['date'] = weather_df['date'].astype(str)
        df = pd.merge(df, weather_df, on=['geohash', 'date'], how='left')
    else:
        print("Warning: Weather features not found. Proceeding without them.")
        
    print("Calculating final Congestion Impact Multiplier (severity_score)...")
    df['severity_score'] = df.apply(calculate_cim, axis=1)
    
    df.to_csv(output_csv, index=False)
    print(f"Feature engineering complete. Saved final dataset to {output_csv}")
    print("You can now export this CSV to Google Colab for GPU training, or run train_model.py locally.")

if __name__ == "__main__":
    merge_and_engineer()
