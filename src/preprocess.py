import pandas as pd
import numpy as np
import os
import urllib.request
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00529/Algerian_forest_fires_dataset_UPDATE.csv"
RAW_DATA_PATH = "data/raw/Algerian_forest_fires_dataset_UPDATE.csv"
PROCESSED_TRAIN_X_PATH = "data/processed/X_train.csv"
PROCESSED_TEST_X_PATH = "data/processed/X_test.csv"
PROCESSED_TRAIN_Y_PATH = "data/processed/y_train.csv"
PROCESSED_TEST_Y_PATH = "data/processed/y_test.csv"
SCALER_PATH = "models/scaler.pkl"

def download_data():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Downloading data from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, RAW_DATA_PATH)
        print("Download complete.")
    else:
        print("Data already exists. Skipping download.")

def preprocess_data():
    print("Loading data...")
    # The dataset has an extra header in row 122 and some missing values
    df = pd.read_csv(RAW_DATA_PATH, header=1)
    
    # Drop rows that are completely empty or have the string "day" (which are headers)
    df.dropna(how='all', inplace=True)
    df = df[~df['day'].astype(str).str.contains('day', na=False, case=False)]
    df = df[~df['day'].astype(str).str.contains('Sidi', na=False, case=False)]
    
    # Strip spaces from column names
    df.columns = df.columns.str.strip()
    
    # Strip spaces from the Classes column
    if 'Classes' in df.columns:
        df['Classes'] = df['Classes'].astype(str).str.strip()
    
    # Create Region column (Bejaia = 0, Sidi Bel-abbes = 1)
    df.reset_index(drop=True, inplace=True)
    df.loc[:121, 'Region'] = 0
    df.loc[122:, 'Region'] = 1
    
    # Convert string columns to numeric
    numeric_cols = ['day', 'month', 'year', 'Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Drop remaining NaNs (if any)
    df.dropna(inplace=True)
    
    # Encode target variable: Fire = 1, not fire = 0
    df['Classes'] = df['Classes'].apply(lambda x: 1 if 'fire' in x and 'not' not in x else 0)
    
    # Features to drop (date columns)
    X = df.drop(['day', 'month', 'year', 'Classes'], axis=1)
    y = df['Classes']
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled features back to DataFrame for saving
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    print("Saving processed data and scaler...")
    X_train_scaled_df.to_csv(PROCESSED_TRAIN_X_PATH, index=False)
    X_test_scaled_df.to_csv(PROCESSED_TEST_X_PATH, index=False)
    y_train.to_csv(PROCESSED_TRAIN_Y_PATH, index=False)
    y_test.to_csv(PROCESSED_TEST_Y_PATH, index=False)
    
    joblib.dump(scaler, SCALER_PATH)
    print("Preprocessing complete! Final data shape:", X.shape)

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    download_data()
    preprocess_data()
