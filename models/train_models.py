import os
import sys
import json
import joblib
import argparse
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models_engine import (
    StandardScalerNumPy,
    LogisticRegressionNumPy,
    DecisionTreeNumPy,
    RandomForestNumPy,
    XGBoostNumPy,
    calculate_metrics
)

FEATURE_COLUMNS = [
    'rainfall_mm',
    'temperature_c',
    'humidity_pct',
    'soil_moisture_pct',
    'slope_deg',
    'elevation_m',
    'previous_rainfall_mm',
    'wind_speed_kmh'
]
TARGET_COLUMN = 'landslide_occurrence'

def train_and_evaluate(data_path="data/landslide_data.csv", output_dir="models", model_filter=None):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset missing at {data_path}")
        
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    
    # Train / Test split (80/20)
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Fit scaler
    scaler = StandardScalerNumPy()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    print("Saved feature scaler to 'models/scaler.pkl'.")
    
    metrics_summary = {
        "dataset_info": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": FEATURE_COLUMNS
        },
        "models": {}
    }
    
    model_configs = {
        "Logistic Regression": {
            "model": LogisticRegressionNumPy(lr=0.05, max_iter=800, random_state=42),
            "scaled": True
        },
        "Decision Tree": {
            "model": DecisionTreeNumPy(max_depth=8, min_samples_split=4, random_state=42),
            "scaled": False
        },
        "Random Forest": {
            "model": RandomForestNumPy(n_trees=25, max_depth=8, min_samples_split=4, random_state=42),
            "scaled": False
        },
        "XGBoost": {
            "model": XGBoostNumPy(n_estimators=30, learning_rate=0.1, max_depth=5, random_state=42),
            "scaled": False
        }
    }

    # Filter to a single model if requested
    if model_filter:
        key_map = {k.lower().replace(" ", "_"): k for k in model_configs}
        if model_filter not in key_map:
            raise ValueError(f"Unknown model '{model_filter}'. Choose from: {list(key_map.keys())}")
        model_configs = {key_map[model_filter]: model_configs[key_map[model_filter]]}

    for name, config in model_configs.items():
        print(f"Training {name}...")

        clf = config["model"]
        is_scaled = config["scaled"]
        
        X_tr = X_train_scaled if is_scaled else X_train
        X_te = X_test_scaled if is_scaled else X_test
        
        clf.fit(X_tr, y_train)
        
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]
        
        m = calculate_metrics(y_test, y_pred, y_prob)
        
        # Feature importances
        if hasattr(clf, 'feature_importances_') and clf.feature_importances_ is not None:
            imp_dict = dict(zip(FEATURE_COLUMNS, np.round(clf.feature_importances_, 4).tolist()))
        else:
            imp_dict = {col: round(1.0/len(FEATURE_COLUMNS), 4) for col in FEATURE_COLUMNS}
            
        m["feature_importance"] = imp_dict
        metrics_summary["models"][name] = m
        
        # Save trained binary
        file_name = f"landslide_{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(clf, os.path.join(output_dir, file_name))
        print(f"  -> {name} trained. Accuracy: {m['accuracy']}, Recall: {m['recall']}, F1: {m['f1_score']}, AUC: {m['roc_auc']}")
        
    metrics_path = os.path.join(output_dir, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    print(f"Metrics summary saved to '{metrics_path}'.")
    return metrics_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Model to train: 'logistic_regression', 'decision_tree', 'random_forest', 'xgboost'")
    args = parser.parse_args()
    train_and_evaluate(model_filter=args.model)
