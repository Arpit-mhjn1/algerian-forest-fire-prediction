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

# Page Config
st.set_page_config(page_title="Forest Fire Prediction", page_icon="🔥", layout="wide")

# Load models and data
@st.cache_resource
def load_models():
    if not os.path.exists("models/model.pkl") or not os.path.exists("models/scaler.pkl"):
        st.info("Models not found. Running data processing and training pipeline on the cloud server... Please wait.")
        from src.preprocess import download_data, preprocess_data
        from src.train import train_and_evaluate
        
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        download_data()
        preprocess_data()
        train_and_evaluate()
        st.success("Training complete!")

    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/raw/Algerian_forest_fires_dataset_UPDATE.csv", header=1)
        df.dropna(how='all', inplace=True)
        df = df[~df['day'].astype(str).str.contains('day', na=False, case=False)]
        df = df[~df['day'].astype(str).str.contains('Sidi', na=False, case=False)]
        df.columns = df.columns.str.strip()
        if 'Classes' in df.columns:
            df['Classes'] = df['Classes'].astype(str).str.strip()
            df['Classes'] = df['Classes'].apply(lambda x: 1 if 'fire' in x and 'not' not in x else 0)
        
        numeric_cols = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading raw data for dashboard: {e}")
        return pd.DataFrame()

# API configuration
def fetch_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=relativehumidity_2m"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            current = data['current_weather']
            # Open-meteo current_weather doesn't always have humidity, so we take the first hourly humidity
            humidity = data['hourly']['relativehumidity_2m'][0] if 'hourly' in data else 50
            return {
                "temp": current['temperature'],
                "wind_speed": current['windspeed'],
                "rain": 0.0, # Approximate or default for current
                "humidity": humidity
            }
    except Exception as e:
        st.error(f"Could not fetch weather: {e}")
    return None

def main():
    st.sidebar.title("🔥 Algerian Forest Fires")
    st.sidebar.markdown("Predict the occurrence of forest fires using machine learning.")
    
    menu = ["Predict", "Dashboard", "Model Evaluation"]
    choice = st.sidebar.selectbox("Navigation", menu)
    
    try:
        model, scaler = load_models()
    except Exception as e:
        st.error("Model or scaler not found. Please run preprocessing and training first.")
        st.stop()
        
    feature_names = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Region']

    if choice == "Predict":
        st.title("Forest Fire Prediction")
        st.markdown("Enter meteorological data to predict the probability of a forest fire.")
        
        # Region selection
        region = st.selectbox("Region", ["Bejaia", "Sidi Bel-abbes"])
        region_val = 0 if region == "Bejaia" else 1
        
        # Live weather button
        if st.button("Fetch Live Weather for Selected Region"):
            lat, lon = (36.75, 5.05) if region == "Bejaia" else (35.2, -0.63)
            weather = fetch_weather(lat, lon)
            if weather:
                st.session_state['temp'] = weather['temp']
                st.session_state['rh'] = weather['humidity']
                st.session_state['ws'] = weather['wind_speed']
                st.session_state['rain'] = weather['rain']
                st.success("Weather data auto-filled!")
                
        # Input Form
        col1, col2 = st.columns(2)
        with col1:
            temp = st.number_input("Temperature (°C)", value=st.session_state.get('temp', 30.0))
            rh = st.number_input("Relative Humidity (%)", value=float(st.session_state.get('rh', 60.0)))
            ws = st.number_input("Wind Speed (km/h)", value=float(st.session_state.get('ws', 15.0)))
            rain = st.number_input("Rain (mm)", value=float(st.session_state.get('rain', 0.0)))
            ffmc = st.number_input("FFMC (Fine Fuel Moisture Code)", value=65.0)
            
        with col2:
            dmc = st.number_input("DMC (Duff Moisture Code)", value=15.0)
            dc = st.number_input("DC (Drought Code)", value=30.0)
            isi = st.number_input("ISI (Initial Spread Index)", value=5.0)
            bui = st.number_input("BUI (Buildup Index)", value=15.0)
            fwi = st.number_input("FWI (Fire Weather Index)", value=5.0)
            
        if st.button("Predict Fire Risk", type="primary"):
            features = np.array([[temp, rh, ws, rain, ffmc, dmc, dc, isi, bui, fwi, region_val]])
            features_scaled = scaler.transform(features)
            
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1] if hasattr(model, 'predict_proba') else prediction
            
            st.markdown("---")
            if prediction == 1:
                st.error(f"🔥 **HIGH RISK OF FIRE** (Probability: {probability*100:.2f}%)")
            else:
                st.success(f"🌲 **LOW RISK OF FIRE** (Probability: {probability*100:.2f}%)")
                
            # Explainability
            st.subheader("Why this prediction?")
            try:
                # TreeExplainer for Tree models, KernelExplainer for others
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(features_scaled)
                # Create a force plot
                shap.initjs()
                
                # Check if shap_values is a list (for some multi-class implementations)
                if isinstance(shap_values, list):
                    shap_val_to_plot = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                else:
                    # Depending on XGBoost version, it might return 2D array for binary classification
                    if len(shap_values.shape) == 2:
                        shap_val_to_plot = shap_values[0]
                    else:
                        shap_val_to_plot = shap_values[1][0] if len(shap_values.shape) > 2 else shap_values[0]
                        
                expected_value = explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

                fig = shap.force_plot(expected_value, shap_val_to_plot, features[0], feature_names=feature_names, matplotlib=True, show=False)
                st.pyplot(fig)
                plt.clf()
            except Exception as e:
                st.warning(f"Could not generate SHAP explanation for the current model type. Error: {e}")

    elif choice == "Dashboard":
        st.title("Data Visualization Dashboard")
        df = load_data()
        
        if not df.empty:
            st.subheader("Fire vs Non-Fire Distribution")
            fig1, ax1 = plt.subplots(figsize=(6,4))
            sns.countplot(data=df, x='Classes', palette='Set2', ax=ax1)
            ax1.set_xticklabels(['Not Fire', 'Fire'])
            st.pyplot(fig1)
            
            st.subheader("Correlation Heatmap")
            numeric_df = df.select_dtypes(include=[np.number])
            fig2, ax2 = plt.subplots(figsize=(10,8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
            st.pyplot(fig2)
            
            st.subheader("Temperature vs FWI")
            fig3, ax3 = plt.subplots(figsize=(8,5))
            sns.scatterplot(data=df, x='Temperature', y='FWI', hue='Classes', palette='Set1', ax=ax3)
            st.pyplot(fig3)
        else:
            st.info("No data available to display.")
            
    elif choice == "Model Evaluation":
        st.title("Model Performance")
        try:
            with open("models/metrics.json", "r") as f:
                metrics = json.load(f)
            
            best_model = metrics.pop("Best_Model", "N/A")
            st.success(f"**Selected Best Model:** {best_model}")
            
            metrics_df = pd.DataFrame(metrics).T
            st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'))
            
            st.bar_chart(metrics_df['F1-Score'])
        except FileNotFoundError:
            st.warning("Metrics file not found. Have you trained the models yet?")

if __name__ == '__main__':
    main()
