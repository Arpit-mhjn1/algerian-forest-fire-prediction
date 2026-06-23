import pandas as pd
import numpy as np
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

PROCESSED_TRAIN_X_PATH = "data/processed/X_train.csv"
PROCESSED_TEST_X_PATH = "data/processed/X_test.csv"
PROCESSED_TRAIN_Y_PATH = "data/processed/y_train.csv"
PROCESSED_TEST_Y_PATH = "data/processed/y_test.csv"
MODEL_PATH = "models/model.pkl"
METRICS_PATH = "models/metrics.json"

def train_and_evaluate():
    print("Loading preprocessed data...")
    X_train = pd.read_csv(PROCESSED_TRAIN_X_PATH)
    X_test = pd.read_csv(PROCESSED_TEST_X_PATH)
    y_train = pd.read_csv(PROCESSED_TRAIN_Y_PATH).values.ravel()
    y_test = pd.read_csv(PROCESSED_TEST_Y_PATH).values.ravel()
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_f1_score = -1
    best_model = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            roc_auc = 0.0
            
        results[name] = {
            "Accuracy": float(accuracy),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1-Score": float(f1),
            "ROC-AUC": float(roc_auc)
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model_name = name
            best_model = model
            
    print(f"\nBest Model: {best_model_name} with F1-Score: {best_f1_score:.4f}")
    
    # Save best model
    print(f"Saving best model ({best_model_name}) to {MODEL_PATH}...")
    joblib.dump(best_model, MODEL_PATH)
    
    # Add best model info to metrics
    results["Best_Model"] = best_model_name
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=4)
        
    print("Training complete!")

if __name__ == "__main__":
    train_and_evaluate()
