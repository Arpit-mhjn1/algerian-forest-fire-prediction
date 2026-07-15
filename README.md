# Algerian Forest Fire Prediction System

An end-to-end, **phone-friendly** and responsive Machine Learning web application to predict the occurrence of forest fires in two regions of Algeria (Bejaia and Sidi Bel-abbes) based on meteorological variables.

## Project Structure

```text
algerian-forest-fire-prediction/
│── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Cleaned & scaled features
│── notebooks/               # Jupyter notebooks for EDA
│── models/                  # Pickled models, scalers, and metrics
│── src/                     # Source code (preprocessing, training)
│── app.py                   # Streamlit web application
│── requirements.txt         # Python dependencies
│── Procfile                 # For deployment on platforms like Render/Heroku
│── README.md                # Project documentation
```

## Setup & Local Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd algerian-forest-fire-prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Processing**:
   Fetch the dataset and run the preprocessing script:
   ```bash
   python src/preprocess.py
   ```

4. **Model Training**:
   Train multiple ML models and pick the best one:
   ```bash
   python src/train.py
   ```

5. **Run the Web Application**:
   ```bash
   streamlit run app.py
   ```

## Features
- **Phone-friendly & Responsive UI**: Premium "Dark Fire" forest aesthetic with custom Streamlit styling, structured card layouts, and complete phone friendliness (optimized touch targets, scaling typography, and responsive margins across phones, tablets, and desktop displays).
- **Interactive Plotly Visualizations**: Features zoomable, interactive, and phone-friendly charts (stacked vertically with horizontal legends for clean viewing on smaller mobile phone screens).
- **Live Weather Integration**: Auto-fills real-time temperature, humidity, wind speed, and rain using the Open-Meteo API.
- **Model Explainability**: Employs SHAP (SHapley Additive exPlanations) force plots to transparently explain individual predictions and feature contributions.
- **Data Visualization Dashboard**: Comprehensive interactive EDA including Fire vs. Non-Fire distribution, Temperature vs. FWI scatter plots, and feature correlation heatmaps.

## Tech Stack
- **Python** (Pandas, NumPy)
- **Scikit-learn & XGBoost** for machine learning models
- **Streamlit** for the frontend dashboard with custom CSS & `.streamlit/config.toml` theming
- **Plotly Express** for interactive, responsive visual charts
- **SHAP** for model explainability
- **Open-Meteo API** for live weather data

## Deployment
This project is ready to be deployed on Streamlit Community Cloud, Hugging Face Spaces, or Render.
- **Streamlit Cloud**: Point to this GitHub repository and select `app.py` as the main script.
- **Render**: The included `Procfile` supports web app deployment (`web: streamlit run app.py --server.port $PORT`). 
