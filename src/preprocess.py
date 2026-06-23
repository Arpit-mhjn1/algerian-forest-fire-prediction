import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

PROCESSED_TRAIN_X_PATH = "data/processed/X_train.csv"
PROCESSED_TEST_X_PATH = "data/processed/X_test.csv"
PROCESSED_TRAIN_Y_PATH = "data/processed/y_train.csv"
PROCESSED_TEST_Y_PATH = "data/processed/y_test.csv"
SCALER_PATH = "models/scaler.pkl"

def download_data():
    pass # Data is fetched directly via ucimlrepo now

def preprocess_data():
    print("Loading data via ucimlrepo...")
    algerian_forest_fires = fetch_ucirepo(id=547)
    df = algerian_forest_fires.data.features.copy()
    df['Classes'] = algerian_forest_fires.data.targets
    
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
    
    # Enforce expected columns strictly
    expected_cols = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Region']
    
    # If there are any other string columns left over (like original 'region'), they are ignored.
    X = df[expected_cols].astype(float)
    y = df['Classes'].astype(int)
    
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
