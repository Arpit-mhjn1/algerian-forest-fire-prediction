# 🔥 Algerian Forest Fire Prediction System 🌲

An end-to-end, **📱 phone-friendly** and responsive Machine Learning web application to predict the occurrence of forest fires in two regions of Algeria (**Bejaia** and **Sidi Bel-abbes**) based on meteorological variables.

---
![Alt text](path/to/your/image.png)

## 📁 Project Structure

```text
algerian-forest-fire-prediction/
│── data/
│   ├── raw/                 # 📂 Original dataset
│   └── processed/           # 🧹 Cleaned & scaled features
│── notebooks/               # 📓 Jupyter notebooks for EDA & prototyping
│── models/                  # 💾 Pickled models, scalers, and evaluation metrics
│── src/                     # ⚙️ Source code (preprocessing, training pipelines)
│── app.py                   # 🚀 Streamlit web application dashboard
│── requirements.txt         # 📦 Python dependencies
│── Procfile                 # ☁️ Deployment config for Render/Heroku
│── README.md                # 📖 Project documentation
```

---

## ⚙️ Setup & Local Installation

1️⃣ **Clone the repository**:
```bash
git clone <repository_url>
cd algerian-forest-fire-prediction
```

2️⃣ **Install dependencies**:
```bash
pip install -r requirements.txt
```

3️⃣ **Data Processing**:
Fetch the dataset and run the automated preprocessing pipeline:
```bash
python src/preprocess.py
```

4️⃣ **Model Training**:
Train multiple machine learning algorithms and automatically select and serialize the best performing model:
```bash
python src/train.py
```

5️⃣ **Run the Web Application**:
Launch the responsive Streamlit dashboard locally:
```bash
streamlit run app.py
```

---

## ✨ Key Features

- 📱 **Phone-friendly & Responsive UI**: Premium *"Dark Fire"* forest aesthetic with custom Streamlit styling, structured card layouts, and complete phone friendliness (optimized touch targets, scaling typography, and responsive margins across phones, tablets, and desktop displays).
- 📊 **Interactive Plotly Visualizations**: Features zoomable, interactive, and phone-friendly charts (stacked vertically with horizontal legends for clean, uncluttered viewing on smaller mobile screens).
- 🌦️ **Live Weather Integration**: Auto-fills real-time temperature, relative humidity, wind speed, and rain using the **Open-Meteo API** with a single click.
- 🧠 **Model Explainability (SHAP)**: Employs **SHAP** *(SHapley Additive exPlanations)* force plots to transparently break down individual predictions and reveal exactly how each feature influenced the fire risk score.
- 📈 **Data Visualization Dashboard**: Comprehensive interactive Exploratory Data Analysis (EDA) including Fire vs. Non-Fire distribution, Temperature vs. FWI scatter plots, and feature correlation heatmaps.

---

## 🛠️ Tech Stack

- 🐍 **Python** *(Pandas, NumPy)* — Data processing and mathematical operations
- 🤖 **Scikit-learn & XGBoost** — High-performance machine learning models
- 🎨 **Streamlit** — Interactive frontend web dashboard with custom CSS & `.streamlit/config.toml` theming
- 📉 **Plotly Express** — Dynamic, responsive, and interactive visual charts
- 🔍 **SHAP** — Deep model explainability and feature attribution
- 🌐 **Open-Meteo API** — Live meteorological data fetching

---

## 🚀 Deployment

This project is ready for instant cloud deployment on platforms like **Streamlit Community Cloud**, **Hugging Face Spaces**, or **Render**:

- ☁️ **Streamlit Cloud**: Simply point to this GitHub repository and select `app.py` as your main script.
- 🌍 **Render**: The included `Procfile` pre-configures web application hosting (`web: streamlit run app.py --server.port $PORT`). 
