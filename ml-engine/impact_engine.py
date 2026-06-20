import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import config

def normalize_score(series):
    """Normalize a pandas Series using Exponential Asymptotic Scaling to [0, 100)."""
    scale_factor = series.std()
    if pd.isna(scale_factor) or scale_factor == 0:
        if series.mean() == 0:
            return pd.Series(0, index=series.index)
        scale_factor = series.mean()
        
    return 100.0 * (1.0 - np.exp(-series / scale_factor))

def calculate_cluster_area(group):
    coords = group[['latitude', 'longitude']].values
    min_area = np.pi * (config.DBSCAN_EPS ** 2)
    if len(coords) < 3:
        # Minimum synthetic area for clusters with <3 points based on EPS radius
        return min_area
    try:
        hull = ConvexHull(coords)
        # In 2D, ConvexHull.volume returns the area, ConvexHull.area returns the perimeter
        return max(hull.volume, min_area)
    except:
        return min_area

def calculate_pii(df):
    print("Calculating Parking Impact Index (PII)...")
    
    total_observation_days = df['created_datetime'].dt.date.nunique()
    if total_observation_days == 0: total_observation_days = 1
    
    # 1. Density Score (using log transform to handle high skew)
    cluster_counts = df.groupby('Cluster_ID').size()
    areas = df.groupby('Cluster_ID').apply(calculate_cluster_area)
    raw_density = cluster_counts / areas
    log_density = np.log1p(raw_density)
    
    # 2. Peak Hour Score
    def is_peak(h):
        return 1 if (8 <= h <= 10) or (17 <= h <= 20) else 0
    df['is_peak'] = df['Hour'].apply(is_peak)
    peak_counts = df.groupby('Cluster_ID')['is_peak'].sum()
    raw_peak = peak_counts / cluster_counts
    
    # 3. Persistence Score
    df['date'] = df['created_datetime'].dt.date
    active_days = df.groupby('Cluster_ID')['date'].nunique()
    raw_persistence = active_days / total_observation_days
    
    # 4. Vehicle Impact Score
    raw_vehicle = df.groupby('Cluster_ID')['Vehicle_Severity'].mean()
    
    # 5. Violation Severity Score
    raw_violation = df.groupby('Cluster_ID')['Violation_Severity'].mean()
    
    # 6. Junction Criticality Score
    def at_junction(val):
        if pd.isna(val) or str(val).strip() == '' or str(val).lower() in ['unknown', 'no junction']:
            return 0
        return 1
    df['is_junction'] = df['junction_name'].apply(at_junction)
    raw_junction = df.groupby('Cluster_ID')['is_junction'].sum() / cluster_counts
    
    # Compile into a DataFrame
    pii_df = pd.DataFrame(index=cluster_counts.index)
    pii_df['Cluster_ID'] = pii_df.index
    pii_df['Total_Violations'] = cluster_counts
    
    # Normalize Component Scores strictly to [0, 100]
    pii_df['Density_Score'] = normalize_score(log_density)
    pii_df['Peak_Hour_Score'] = normalize_score(raw_peak)
    pii_df['Persistence_Score'] = normalize_score(raw_persistence)
    pii_df['Vehicle_Impact_Score'] = normalize_score(raw_vehicle)
    pii_df['Violation_Severity_Score'] = normalize_score(raw_violation)
    pii_df['Junction_Criticality_Score'] = normalize_score(raw_junction)
    
    # Final PII Calculation (weighted sum, then normalized to exactly [0, 100])
    raw_pii = (
        (0.30 * pii_df['Density_Score']) +
        (0.20 * pii_df['Peak_Hour_Score']) +
        (0.15 * pii_df['Persistence_Score']) +
        (0.15 * pii_df['Vehicle_Impact_Score']) +
        (0.10 * pii_df['Violation_Severity_Score']) +
        (0.10 * pii_df['Junction_Criticality_Score'])
    )
    pii_df['PII'] = normalize_score(raw_pii)
    
    # Severity Classification
    def get_severity(score):
        if score <= 25: return 'Low'
        if score <= 50: return 'Medium'
        if score <= 75: return 'High'
        return 'Critical'
        
    pii_df['Severity_Tier'] = pii_df['PII'].apply(get_severity)
    
    # Save the impact scores
    pii_df = pii_df.sort_values(by='PII', ascending=False)
    pii_df.to_csv(config.IMPACT_SCORES_PATH, index=False)
    print("PII Calculation Complete and saved.")
    
    return pii_df

def run_impact_engine():
    try:
        df = pd.read_csv(config.PROCESSED_DATA_PATH)
        df['created_datetime'] = pd.to_datetime(df['created_datetime'])
        calculate_pii(df)
    except FileNotFoundError:
        print("Processed data not found. Run data_pipeline.py first.")

if __name__ == "__main__":
    run_impact_engine()
