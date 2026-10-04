import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Ensure root import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models_engine import (
    StandardScalerNumPy,
    LogisticRegressionNumPy,
    DecisionTreeNumPy,
    RandomForestNumPy,
    XGBoostNumPy
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

FEATURE_LABELS = {
    'rainfall_mm': '24h Rainfall (mm)',
    'temperature_c': 'Temperature (°C)',
    'humidity_pct': 'Relative Humidity (%)',
    'soil_moisture_pct': 'Soil Moisture (%)',
    'slope_deg': 'Slope Gradient (°)',
    'elevation_m': 'Elevation (m)',
    'previous_rainfall_mm': 'Antecedent Rainfall (mm)',
    'wind_speed_kmh': 'Wind Speed (km/h)'
}

class LandslidePredictor:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.scaler = None
        self.models = {}
        self.metrics = {}
        self._load_artifacts()

    def _load_artifacts(self):
        # Load Scaler
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        # Load Metrics
        metrics_path = os.path.join(self.models_dir, "model_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)

        # Load Models
        model_names = ["Random Forest", "Logistic Regression", "Decision Tree", "XGBoost"]
        for name in model_names:
            file_name = f"landslide_{name.lower().replace(' ', '_')}_model.pkl"
            full_path = os.path.join(self.models_dir, file_name)
            if os.path.exists(full_path):
                self.models[name] = joblib.load(full_path)

    def predict(self, input_dict, model_name="Random Forest"):
        """
        Runs prediction for a single instance of environmental values.
        input_dict keys: FEATURE_COLUMNS
        """
        if model_name not in self.models:
            model_name = "Random Forest"

        model = self.models[model_name]
        
        # Construct feature vector
        raw_vals = [float(input_dict.get(col, 0.0)) for col in FEATURE_COLUMNS]
        X_raw = np.array([raw_vals])

        # Apply scaling if model requires it
        if model_name == "Logistic Regression" and self.scaler is not None:
            X_in = self.scaler.transform(X_raw)
        else:
            X_in = X_raw

        # Predict probability
        proba_pair = model.predict_proba(X_in)[0]
        prob_landslide = float(proba_pair[1])
        prob_pct = round(prob_landslide * 100, 1)

        # Risk Classification Thresholds
        if prob_landslide < 0.35:
            risk_level = "Low Risk"
            risk_code = "LOW"
            badge_color = "#10b981" # Green
            advice = "Environmental conditions are stable. Standard monitoring recommended."
        elif prob_landslide <= 0.70:
            risk_level = "Medium Risk"
            risk_code = "MEDIUM"
            badge_color = "#f59e0b" # Orange
            advice = "Moderate hazard potential. Soil moisture or slope gradient elevated. Stay alert."
        else:
            risk_level = "High Risk"
            risk_code = "HIGH"
            badge_color = "#ef4444" # Red
            advice = "CRITICAL HAZARD WARNING! High landslide probability. Issue immediate public advisory/evacuation alert!"

        # Instance-specific contributing factors breakdown
        feat_importances = getattr(model, 'feature_importances_', None)
        if feat_importances is None:
            feat_importances = np.ones(len(FEATURE_COLUMNS)) / len(FEATURE_COLUMNS)

        # Calculate instance impact: weight * normalized_value
        norm_vals = np.abs(raw_vals) / (np.max(np.abs(raw_vals)) + 1e-5)
        impacts = feat_importances * (norm_vals + 0.1)
        impacts = impacts / np.sum(impacts)

        factors = []
        for i, col in enumerate(FEATURE_COLUMNS):
            factors.append({
                "feature": col,
                "label": FEATURE_LABELS[col],
                "val": raw_vals[i],
                "impact_pct": round(float(impacts[i]) * 100, 1),
                "global_importance": round(float(feat_importances[i]) * 100, 1)
            })

        # Sort factors by impact
        factors = sorted(factors, key=lambda x: x["impact_pct"], reverse=True)

        return {
            "model_used": model_name,
            "probability": prob_landslide,
            "probability_pct": prob_pct,
            "risk_level": risk_level,
            "risk_code": risk_code,
            "badge_color": badge_color,
            "advice": advice,
            "contributing_factors": factors,
            "input_values": dict(zip(FEATURE_COLUMNS, raw_vals))
        }
