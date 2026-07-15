import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import plotly.express as px
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title="Forest Fire Prediction", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")

# Load models and data
@st.cache_resource
def load_models():
    if not os.path.exists("models/model.pkl") or not os.path.exists("models/scaler.pkl"):
        with st.spinner("Setting up prediction engine and training models..."):
            from src.preprocess import download_data, preprocess_data
            from src.train import train_and_evaluate
            
            os.makedirs("data/raw", exist_ok=True)
            os.makedirs("data/processed", exist_ok=True)
            os.makedirs("models", exist_ok=True)
            
            download_data()
            preprocess_data()
            train_and_evaluate()

    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

@st.cache_data
def load_data():
    try:
        from ucimlrepo import fetch_ucirepo
        algerian_forest_fires = fetch_ucirepo(id=547) 
        df = algerian_forest_fires.data.features.copy()
        
        # safely assign targets
        targets = algerian_forest_fires.data.targets
        df['Classes'] = targets.iloc[:, 0] if isinstance(targets, pd.DataFrame) else targets
        
        df.columns = df.columns.str.strip()
        
        numeric_cols = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        
        if 'Classes' in df.columns:
            df['Classes'] = df['Classes'].apply(lambda x: 1 if isinstance(x, str) and 'fire' in x.lower() and 'not' not in x.lower() else 0)
            
        return df
    except Exception as e:
        st.error(f"Error loading raw data for dashboard: {str(e)}")
        return pd.DataFrame()

# API configuration
def fetch_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=relativehumidity_2m"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            current = data['current_weather']
            humidity = data['hourly']['relativehumidity_2m'][0] if 'hourly' in data else 50
            return {
                "temp": current['temperature'],
                "wind_speed": current['windspeed'],
                "rain": 0.0,
                "humidity": humidity
            }
    except Exception as e:
        st.error(f"Could not fetch weather: {e}")
    return None

def main():
    # Inject CSS for backgrounds and overall styling
    st.markdown(
        """
        <style>
        /* Main App Background */
        .stApp {
            background: linear-gradient(rgba(10, 15, 12, 0.8), rgba(10, 15, 12, 0.8)), url("https://images.unsplash.com/photo-1448375240586-882707db888b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        
        /* Clean Top Navigation Tab Bar */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background-color: transparent !important;
            padding: 0px !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.15) !important;
        }
        
        /* Streamlit highlight line styling */
        .stTabs [data-baseweb="tab-border"] {
            display: none !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #ff4b4b !important;
            height: 3px !important;
        }
        
        /* Individual Tabs - No Background Box, Minimal Padding */
        .stTabs [data-baseweb="tab"],
        .stTabs [data-baseweb="tab"] *,
        .stTabs [data-baseweb="tab"] p,
        .stTabs button[role="tab"],
        .stTabs button[role="tab"] * {
            height: auto !important;
            min-height: 54px !important;
            white-space: pre-wrap;
            background-color: transparent !important;
            border-radius: 0px !important;
            padding: 8px 8px !important;
            color: #cdd5d0 !important;
            font-weight: 600 !important;
            font-size: 1.6rem !important;
            border: none !important;
            transition: all 0.2s ease-in-out;
        }
        
        /* Tab Hover Effect */
        .stTabs [data-baseweb="tab"]:hover,
        .stTabs [data-baseweb="tab"]:hover *,
        .stTabs button[role="tab"]:hover * {
            background-color: transparent !important;
            color: #ffffff !important;
        }
        
        /* Active Selected Tab - Transparent Background, Bright White Text */
        .stTabs [aria-selected="true"],
        .stTabs [aria-selected="true"] *,
        .stTabs [aria-selected="true"] p,
        .stTabs button[role="tab"][aria-selected="true"],
        .stTabs button[role="tab"][aria-selected="true"] * {
            background-color: transparent !important;
            color: #ffffff !important;
            font-weight: 800 !important;
            font-size: 1.6rem !important;
            box-shadow: none !important;
            border-radius: 0px !important;
            transform: none !important;
        }
        /* Keep dropdown menu text dark */
        div[role="listbox"] span {
            color: black !important;
            text-shadow: none;
        }
        
        /* Stylized prediction box */
        .pred-box {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .high-risk {
            background-color: rgba(255, 75, 75, 0.2);
            border: 2px solid #ff4b4b;
        }
        .low-risk {
            background-color: rgba(9, 171, 59, 0.2);
            border: 2px solid #09ab3b;
        }
        
        /* Responsive Mobile Styles */
        @media screen and (max-width: 768px) {
            .stApp {
                background-attachment: scroll !important;
                background-size: cover !important;
            }
            .block-container {
                padding-top: 1.5rem !important;
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
            }
            .pred-box {
                padding: 12px !important;
            }
            .pred-box h2 {
                font-size: 1.3rem !important;
            }
            .pred-box h3 {
                font-size: 1.05rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    try:
        model, scaler = load_models()
    except Exception as e:
        st.error(f"Error during training pipeline or model loading: {str(e)}")
        st.stop()
        
    feature_names = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Region']

    tab_predict, tab_dashboard, tab_eval = st.tabs(["Predict", "Dashboard", "Model Evaluation"])

    with tab_predict:
        st.title("Algerian Forest Fire Prediction System")
        st.markdown("Predict the occurrence of forest fires using machine learning and explore meteorological data.")
        st.divider()
        st.subheader("Forest Fire Prediction Engine")
        st.markdown("Enter meteorological data and FWI indices below to predict the probability of a forest fire.")
        
        # Region selection & Live weather button in a row
        col_reg, col_btn = st.columns([1, 2])
        with col_reg:
            region = st.selectbox("Region", ["Bejaia", "Sidi Bel-abbes"])
            region_val = 0 if region == "Bejaia" else 1
            
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True) # align button
            if st.button("Fetch Live Weather for Selected Region", use_container_width=True):
                lat, lon = (36.75, 5.05) if region == "Bejaia" else (35.2, -0.63)
                weather = fetch_weather(lat, lon)
                if weather:
                    st.session_state['temp'] = weather['temp']
                    st.session_state['rh'] = weather['humidity']
                    st.session_state['ws'] = weather['wind_speed']
                    st.session_state['rain'] = weather['rain']
                    st.success(f"Weather data auto-filled for {region}!")
                    
        st.divider()

        # Input Form grouped in native container cards
        st.subheader("Feature Inputs")
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("#### Meteorological Data")
                temp = st.number_input("Temperature (°C)", value=st.session_state.get('temp', 30.0))
                rh = st.number_input("Relative Humidity (%)", value=float(st.session_state.get('rh', 60.0)))
                ws = st.number_input("Wind Speed (km/h)", value=float(st.session_state.get('ws', 15.0)))
                rain = st.number_input("Rain (mm)", value=float(st.session_state.get('rain', 0.0)))
                ffmc = st.number_input("FFMC (Fine Fuel Moisture Code)", value=65.0)
            
        with col2:
            with st.container(border=True):
                st.markdown("#### FWI System Components")
                dmc = st.number_input("DMC (Duff Moisture Code)", value=15.0)
                dc = st.number_input("DC (Drought Code)", value=30.0)
                isi = st.number_input("ISI (Initial Spread Index)", value=5.0)
                bui = st.number_input("BUI (Buildup Index)", value=15.0)
                fwi = st.number_input("FWI (Fire Weather Index)", value=5.0)
            
        if st.button("Predict Fire Risk", type="primary", use_container_width=True):
            features = np.array([[temp, rh, ws, rain, ffmc, dmc, dc, isi, bui, fwi, region_val]])
            features_scaled = scaler.transform(features)
            
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1] if hasattr(model, 'predict_proba') else prediction
            
            if prediction == 1:
                st.markdown(f'<div class="pred-box high-risk"><h2>HIGH RISK OF FIRE</h2><h3>Probability: {probability*100:.1f}%</h3></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="pred-box low-risk"><h2>LOW RISK OF FIRE</h2><h3>Probability: {probability*100:.1f}%</h3></div>', unsafe_allow_html=True)
                
            # Explainability
            st.subheader("Why this prediction?")
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(features_scaled)
                
                # Safely extract SHAP values for the single instance (class 1 if multi-class)
                if isinstance(shap_values, list):
                    shap_val_to_plot = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                elif hasattr(shap_values, 'shape'):
                    if len(shap_values.shape) == 3:
                        # shape could be (samples, features, classes) -> (1, 11, 2)
                        if shap_values.shape[2] >= 2:
                            shap_val_to_plot = shap_values[0, :, 1]
                        # shape could be (samples, classes, features) -> (1, 2, 11)
                        elif shap_values.shape[1] >= 2:
                            shap_val_to_plot = shap_values[0, 1, :]
                        else:
                            shap_val_to_plot = shap_values[0, 0, :]
                    elif len(shap_values.shape) == 2:
                        shap_val_to_plot = shap_values[0]
                    else:
                        shap_val_to_plot = shap_values
                else:
                    shap_val_to_plot = shap_values[0]
                        
                expected_value = explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

                # We use pyplot for SHAP since it renders best natively
                fig = shap.force_plot(expected_value, shap_val_to_plot, features[0], feature_names=feature_names, matplotlib=True, show=False)
                st.pyplot(fig, use_container_width=True)
                plt.clf()
            except Exception as e:
                st.warning(f"Could not generate SHAP explanation for the current model type. Error: {e}")

    with tab_dashboard:
        st.title("Algerian Forest Fire Prediction System")
        st.markdown("Explore and visualize dataset distributions, trends, and variable correlations.")
        st.divider()
        st.subheader("Data Visualization Dashboard")
        df = load_data()
        
        if not df.empty:
            df['Classes_Label'] = df['Classes'].map({0: 'Not Fire', 1: 'Fire'})
            
            st.subheader("Fire vs Non-Fire Distribution")
            fig_dist = px.pie(df, names='Classes_Label', color='Classes_Label', 
                              color_discrete_map={'Not Fire':'#09ab3b', 'Fire':'#ff4b4b'},
                              hole=0.4)
            fig_dist.update_traces(textinfo='percent+label', textfont_size=14)
            fig_dist.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                margin=dict(t=30, b=30, l=10, r=10)
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            st.divider()
            
            st.subheader("Temperature vs FWI")
            fig_scatter = px.scatter(df, x='Temperature', y='FWI', color='Classes_Label',
                                     color_discrete_map={'Not Fire':'#09ab3b', 'Fire':'#ff4b4b'},
                                     size='Temperature', hover_data=['RH', 'Ws'])
            fig_scatter.update_layout(
                legend_title_text='Status',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=40, b=20, l=10, r=10)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
                
            st.divider()
            
            st.subheader("Correlation Heatmap")
            numeric_df = df.select_dtypes(include=[np.number])
            corr = numeric_df.corr()
            fig_heatmap = px.imshow(corr, text_auto=".2f", aspect="auto", 
                                    color_continuous_scale='RdBu_r')
            fig_heatmap.update_layout(margin=dict(t=20, b=20, l=10, r=10))
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("No data available to display.")
            
    with tab_eval:
        st.title("Algerian Forest Fire Prediction System")
        st.markdown("Review machine learning model metrics, comparisons, and performance benchmarks.")
        st.divider()
        st.subheader("Model Performance")
        try:
            with open("models/metrics.json", "r") as f:
                metrics = json.load(f)
            
            best_model = metrics.pop("Best_Model", "N/A")
            st.info(f"**Selected Best Model:** {best_model}")
            
            metrics_df = pd.DataFrame(metrics).T
            
            st.markdown("#### F1-Score Comparison")
            fig_eval = px.bar(metrics_df, x=metrics_df.index, y='F1-Score', color='F1-Score',
                              color_continuous_scale='YlOrRd')
            fig_eval.update_layout(
                xaxis_title="Model", 
                yaxis_title="F1-Score",
                margin=dict(t=30, b=20, l=10, r=10)
            )
            st.plotly_chart(fig_eval, use_container_width=True)
            
            st.divider()
            
            st.markdown("#### Metrics Table")
            st.dataframe(metrics_df.style.highlight_max(axis=0, color='#ff4b4b'), use_container_width=True)
                
        except FileNotFoundError:
            st.warning("Metrics file not found. Have you trained the models yet?")

if __name__ == '__main__':
    main()
