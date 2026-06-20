import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import joblib
import config

st.set_page_config(page_title="AI Parking Intelligence Platform", layout="wide", initial_sidebar_state="expanded")

# Inject Custom Premium CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Font Family */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Premium Gradient Headers */
    .title-container {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8C00 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.2);
    }
    
    .title-text {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        color: white !important;
    }
    
    .subtitle-text {
        font-size: 1.1rem;
        font-weight: 300;
        margin: 5px 0 0 0;
        opacity: 0.9;
    }
    
    /* KPI Cards */
    .card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #eaeaea;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        margin-bottom: 20px;
    }
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    /* Dark mode adjustments for card text if theme is dark */
    @media (prefers-color-scheme: dark) {
        .card {
            background-color: #1e1e1e;
            border-color: #333333;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data
def load_impact_data():
    try:
        df = pd.read_csv(config.IMPACT_SCORES_PATH)
        centroids = pd.read_csv(config.CLUSTER_GEO_PATH)
        df = df.merge(centroids, on='Cluster_ID', how='left')
        return df
    except:
        return pd.DataFrame()

@st.cache_data
def load_raw_data():
    try:
        return pd.read_csv(config.RAW_DATA_PATH)
    except:
        return pd.DataFrame()
        
@st.cache_resource
def load_model():
    try:
        return joblib.load(config.MODEL_SAVE_PATH)
    except:
        return None

# --- Main App ---
st.markdown("""
<div class="title-container">
    <h1 class="title-text">🚦 AI-Driven Parking Intelligence Platform</h1>
    <p class="subtitle-text">Dual-Engine Architecture: Impact Assessment & Violation Prediction</p>
</div>
""", unsafe_allow_html=True)

impact_data = load_impact_data()
model_data = load_model()

tab1, tab2 = st.tabs(["📊 Impact Panel (Current Operations)", "🔮 Prediction Panel (Future Operations)"])

# ====== TAB 1: IMPACT PANEL ======
with tab1:
    st.header("Active Enforcement Priority Rankings")
    
    if not impact_data.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Spatial Distribution of PII Hotspots")
            
            # Folium Map
            center_lat = impact_data['latitude'].mean() if not impact_data['latitude'].isna().all() else 0
            center_lon = impact_data['longitude'].mean() if not impact_data['longitude'].isna().all() else 0
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")
            
            color_map = {'Critical': 'red', 'High': 'orange', 'Medium': 'yellow', 'Low': 'green'}
            
            for idx, row in impact_data.iterrows():
                if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=max(5, (row['Total_Violations'] / impact_data['Total_Violations'].max()) * 20),
                        color=color_map.get(row['Severity_Tier'], 'gray'),
                        fill=True,
                        fill_opacity=0.7,
                        tooltip=f"Cluster {row['Cluster_ID']} - PII: {row['PII']:.1f} ({row['Severity_Tier']})"
                    ).add_to(m)
            
            folium_static(m, width=700, height=500)
            
        with col2:
            st.subheader("Top 5 Clusters Analysis")
            top_5 = impact_data.head(5)
            
            # Bar chart for Vehicle Impact & Violation Severity
            fig = go.Figure()
            fig.add_trace(go.Bar(x=top_5['Cluster_ID'].astype(str), y=top_5['Vehicle_Impact_Score'], name='Vehicle Impact'))
            fig.add_trace(go.Bar(x=top_5['Cluster_ID'].astype(str), y=top_5['Violation_Severity_Score'], name='Violation Severity'))
            fig.update_layout(barmode='group', xaxis_title="Cluster ID", yaxis_title="Score (0-100)", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Enforcement Priority Ranking Data")
        display_cols = ['Cluster_ID', 'PII', 'Severity_Tier', 'Total_Violations', 'Density_Score', 'Peak_Hour_Score', 'Persistence_Score', 'Vehicle_Impact_Score', 'Violation_Severity_Score', 'Junction_Criticality_Score']
        st.dataframe(impact_data[display_cols].style.background_gradient(subset=['PII'], cmap='Reds'))
        
    else:
        st.warning("Impact data not found. Please run the Impact Engine pipeline.")

# ====== TAB 2: PREDICTION PANEL ======
with tab2:
    st.header("Future Violation Hotspot Forecast")
    
    if model_data and not impact_data.empty:
        # Beautiful KPI Metrics for Model Performance
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f"""
            <div class="card">
                <div style="font-size:0.9rem;color:#7f8c8d;font-weight:600;">ACTIVE ML MODEL</div>
                <div style="font-size:1.8rem;font-weight:800;color:#2c3e50;">{model_data['model_name']}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="card">
                <div style="font-size:0.9rem;color:#7f8c8d;font-weight:600;">MODEL R² SCORE</div>
                <div style="font-size:1.8rem;font-weight:800;color:#2ecc71;">{model_data['r2']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="card">
                <div style="font-size:0.9rem;color:#7f8c8d;font-weight:600;">ROOT MEAN SQUARED ERROR (RMSE)</div>
                <div style="font-size:1.8rem;font-weight:800;color:#e74c3c;">{model_data['rmse']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # User Inputs
        col_input1, col_input2, col_input3 = st.columns(3)
        with col_input1:
            selected_day = st.selectbox("Select Future Day of Week", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])
        with col_input2:
            selected_hour = st.slider("Select Future Hour", min_value=0, max_value=23, value=12)
        with col_input3:
            selected_month = st.selectbox("Select Month Context", options=list(range(1, 13)))
            
        is_weekend = 1 if selected_day >= 5 else 0
        
        if st.button("Generate Forecast"):
            with st.spinner("Forecasting violation counts across all spatial clusters..."):
                model = model_data['model']
                features = model_data['features']
                
                # We need the advanced temporal features (lags, density) from the aggregated dataset
                try:
                    agg_df = pd.read_csv(config.AGGREGATED_DATA_PATH)
                    # Get the most recent row for each cluster to use its lags as the 'current' state
                    latest_lags = agg_df.sort_values('Date_Hour').groupby('Cluster_ID').tail(1)[
                        ['Cluster_ID', 'cluster_centroid_lat', 'cluster_centroid_lon',
                         'violations_lag_1h', 'rolling_3h_mean', 
                         'avg_vehicle_weight', 'antigravity_repulsion_factor']
                    ]
                    inference_df = pd.DataFrame({'Cluster_ID': impact_data['Cluster_ID']})
                    inference_df = inference_df.merge(latest_lags, on='Cluster_ID', how='left')
                except Exception as e:
                    st.error(f"Error loading aggregated data: {e}")
                    inference_df = pd.DataFrame()
                
                # Add Cyclical Time Encoding based on user selection
                inference_df['hour_sin'] = np.sin(2 * np.pi * selected_hour / 24)
                inference_df['hour_cos'] = np.cos(2 * np.pi * selected_hour / 24)
                
                # Merge the text features from impact_data for interpretability
                inference_df = inference_df.merge(
                    impact_data[['Cluster_ID', 'violation_type', 'police_station']], 
                    on='Cluster_ID', 
                    how='left'
                )
                
                # Fallback to zero for any NaNs
                inference_df = inference_df.fillna(0)
                
                # Predict
                X_infer = inference_df[features]
                preds = model.predict(X_infer)
                
                inference_df['Target_Severity'] = np.clip(np.round(preds, 2), 0, 100)
                
                # Hotspot Categorization based on 0-100 score
                def get_forecast_tier(score):
                    if score <= 25: return "Low"
                    if score <= 50: return "Medium"
                    if score <= 80: return "High"
                    return "Critical"
                
                inference_df['Forecast_Category'] = inference_df['Target_Severity'].apply(get_forecast_tier)
                
                st.subheader(f"Risk Map Forecast (Day {selected_day}, {selected_hour}:00)")
                
                # Ensure latitude/longitude exist for plotly
                inference_df['latitude'] = inference_df['cluster_centroid_lat']
                inference_df['longitude'] = inference_df['cluster_centroid_lon']
                
                # Plotly Map
                fig_map = px.scatter_mapbox(
                    inference_df, 
                    lat="latitude", 
                    lon="longitude", 
                    color="Forecast_Category",
                    size="Target_Severity",
                    color_discrete_map={'Critical': 'red', 'High': 'orange', 'Medium': 'yellow', 'Low': 'green'},
                    hover_name="Cluster_ID",
                    hover_data=["Target_Severity", "violation_type", "police_station"],
                    mapbox_style="carto-positron",
                    zoom=11
                )
                fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig_map, use_container_width=True)
                
                st.dataframe(inference_df[['Cluster_ID', 'Target_Severity', 'Forecast_Category', 'violation_type', 'police_station']].sort_values(by='Target_Severity', ascending=False))
                
    else:
        st.warning("Model or Cluster data not found. Please run the model training pipeline.")
