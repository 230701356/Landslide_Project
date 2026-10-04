import sys
import os

# Append project root
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.predictor import LandslidePredictor

def run_model_cli(model_name="Random Forest"):
    # Check command line arguments for model selection
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if "xgboost" in arg:
            model_name = "XGBoost"
        elif "decision" in arg or "tree" in arg:
            model_name = "Decision Tree"
        elif "logistic" in arg or "regression" in arg:
            model_name = "Logistic Regression"
        elif "forest" in arg or "rf" in arg:
            model_name = "Random Forest"

    print("=" * 65)
    print(f"  LANDSLIDE PREDICTION SYSTEM -- {model_name.upper()} MODEL (CLI)")
    print("=" * 65)
    
    predictor = LandslidePredictor(models_dir="models")
    
    # Check if models are trained
    if model_name not in predictor.models:
        print(f"\nTrained model '{model_name}' not found! Training models now...\n")
        from models.train_models import train_and_evaluate
        train_and_evaluate()
        predictor = LandslidePredictor(models_dir="models")

    # Display Model Metrics Summary
    metrics = predictor.metrics.get("models", {}).get(model_name, {})
    if metrics:
        print(f"\n{model_name.upper()} MODEL BENCHMARK METRICS:")
        print(f"   * Test Accuracy  : {metrics.get('accuracy', 0)*100:.2f}%")
        print(f"   * Precision      : {metrics.get('precision', 0)*100:.2f}%")
        print(f"   * Recall         : {metrics.get('recall', 0)*100:.2f}%")
        print(f"   * F1-Score       : {metrics.get('f1_score', 0)*100:.2f}%")
        print(f"   * ROC-AUC Score  : {metrics.get('roc_auc', 0):.4f}")
    print("-" * 65)

    # Test Scenarios
    test_cases = [
        {
            "title": "TEST CASE 1: Heavy Monsoon Storm (High Hazard)",
            "data": {
                'rainfall_mm': 180.0,
                'temperature_c': 19.0,
                'humidity_pct': 88.0,
                'soil_moisture_pct': 74.0,
                'slope_deg': 38.0,
                'elevation_m': 950.0,
                'previous_rainfall_mm': 140.0,
                'wind_speed_kmh': 45.0
            }
        },
        {
            "title": "TEST CASE 2: Moderate Rain & Medium Slope (Medium Risk)",
            "data": {
                'rainfall_mm': 65.0,
                'temperature_c': 22.0,
                'humidity_pct': 72.0,
                'soil_moisture_pct': 52.0,
                'slope_deg': 24.0,
                'elevation_m': 600.0,
                'previous_rainfall_mm': 45.0,
                'wind_speed_kmh': 20.0
            }
        },
        {
            "title": "TEST CASE 3: Fair Weather & Dry Soil (Low Risk / Safe)",
            "data": {
                'rainfall_mm': 10.0,
                'temperature_c': 26.0,
                'humidity_pct': 45.0,
                'soil_moisture_pct': 20.0,
                'slope_deg': 12.0,
                'elevation_m': 300.0,
                'previous_rainfall_mm': 5.0,
                'wind_speed_kmh': 10.0
            }
        }
    ]

    for test in test_cases:
        print(f"\n[+] {test['title']}")
        res = predictor.predict(test['data'], model_name=model_name)
        
        print(f"    - Input Readings : Rain={test['data']['rainfall_mm']}mm, SoilMoisture={test['data']['soil_moisture_pct']}%, Slope={test['data']['slope_deg']}deg, PrevRain={test['data']['previous_rainfall_mm']}mm")
        print(f"    - Predicted Prob : {res['probability_pct']}%")
        print(f"    - Risk Category  : [{res['risk_level'].upper()}]")
        print(f"    - Advisory       : {res['advice']}")
        
        # Show top contributing factors
        factors = res['contributing_factors'][:3]
        top_factors_str = ", ".join([f"{f['label']}={f['val']}" for f in factors])
        print(f"    - Top Triggers   : {top_factors_str}")

    print("\n" + "=" * 65)

if __name__ == "__main__":
    run_model_cli()
