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
st.title("🚦 AI-Driven Parking Intelligence Platform")
st.markdown("Dual-Engine Architecture: Impact Assessment & Violation Prediction")

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
        st.success(f"**Active Model:** {model_data['model_name']} | **R² Score:** {model_data['r2']:.4f} | **RMSE:** {model_data['rmse']:.4f}")
        
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
                
                # Build inference dataframe for all clusters
                inference_df = pd.DataFrame({'Cluster_ID': impact_data['Cluster_ID']})
                inference_df['Hour'] = selected_hour
                inference_df['DayOfWeek'] = selected_day
                inference_df['Month'] = selected_month
                inference_df['Weekend_Flag'] = is_weekend
                
                # Mock average severity and junction info for inference based on historical means
                inference_df['Vehicle_Severity'] = config.DEFAULT_VEHICLE_WEIGHT
                inference_df['Violation_Severity'] = config.DEFAULT_VIOLATION_WEIGHT
                inference_df['junction_encoded'] = 0
                inference_df['police_encoded'] = 0
                
                # Predict
                X_infer = inference_df[features]
                preds = model.predict(X_infer)
                
                inference_df['Predicted_Count'] = np.maximum(0, np.round(preds))
                
                # Hotspot Categorization
                def get_forecast_tier(cnt):
                    if cnt <= 10: return "Low"
                    if cnt <= 25: return "Medium"
                    if cnt <= 50: return "High"
                    return "Critical"
                
                inference_df['Forecast_Category'] = inference_df['Predicted_Count'].apply(get_forecast_tier)
                
                # Merge coordinates
                inference_df = inference_df.merge(impact_data[['Cluster_ID', 'latitude', 'longitude']], on='Cluster_ID', how='left')
                
                st.subheader(f"Risk Map Forecast (Day {selected_day}, {selected_hour}:00)")
                
                # Plotly Map
                fig_map = px.scatter_mapbox(
                    inference_df, 
                    lat="latitude", 
                    lon="longitude", 
                    color="Forecast_Category",
                    size="Predicted_Count",
                    color_discrete_map={'Critical': 'red', 'High': 'orange', 'Medium': 'yellow', 'Low': 'green'},
                    hover_name="Cluster_ID",
                    hover_data=["Predicted_Count"],
                    mapbox_style="carto-positron",
                    zoom=11
                )
                fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig_map, use_container_width=True)
                
                st.dataframe(inference_df[['Cluster_ID', 'Predicted_Count', 'Forecast_Category']].sort_values(by='Predicted_Count', ascending=False))
                
    else:
        st.warning("Model or Cluster data not found. Please run the model training pipeline.")
