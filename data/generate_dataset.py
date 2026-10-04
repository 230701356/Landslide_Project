import os
import numpy as np
import pandas as pd

def generate_landslide_dataset(n_samples=2500, random_seed=42, output_path="data/landslide_data.csv"):
    """
    Generates a realistic synthetic historical dataset for landslide hazard modeling
    based on environmental and geographical parameters.
    """
    np.random.seed(random_seed)
    
    # 1. Feature Generation with realistic distributions
    rainfall_mm = np.random.gamma(shape=2.5, scale=25.0, size=n_samples) # 0 - 300 mm
    rainfall_mm = np.clip(rainfall_mm, 0, 350)
    
    slope_deg = np.random.uniform(5, 55, size=n_samples) # 5 to 55 degrees
    
    # Soil moisture is correlated with rainfall
    soil_moisture_pct = np.clip(
        0.3 * rainfall_mm + np.random.normal(loc=35, scale=15, size=n_samples),
        10, 98
    )
    
    previous_rainfall_mm = np.random.gamma(shape=2.0, scale=20.0, size=n_samples)
    previous_rainfall_mm = np.clip(previous_rainfall_mm, 0, 300)
    
    humidity_pct = np.clip(
        0.2 * rainfall_mm + np.random.normal(loc=60, scale=12, size=n_samples),
        30, 100
    )
    
    temperature_c = np.random.normal(loc=22, scale=6, size=n_samples)
    temperature_c = np.clip(temperature_c, 5, 42)
    
    elevation_m = np.random.uniform(100, 2800, size=n_samples)
    
    wind_speed_kmh = np.random.weibull(a=1.8, size=n_samples) * 15
    wind_speed_kmh = np.clip(wind_speed_kmh, 2, 85)
    
    # 2. Physics-informed Landslide Probability Formula (Logit model with noise)
    # Landslides primarily depend on Slope + Soil Moisture + 24h Rainfall + Antecedent Rainfall
    z = (
        -7.2 
        + 0.022 * rainfall_mm 
        + 0.038 * soil_moisture_pct 
        + 0.095 * slope_deg 
        + 0.018 * previous_rainfall_mm 
        + 0.012 * humidity_pct 
        + 0.0003 * elevation_m 
        - 0.025 * temperature_c 
        + np.random.normal(0, 0.45, size=n_samples) # Realistic noise/stochasticity
    )
    
    prob = 1.0 / (1.0 + np.exp(-z))
    landslide_occurrence = (prob >= 0.48).astype(int)
    
    df = pd.DataFrame({
        'rainfall_mm': np.round(rainfall_mm, 2),
        'temperature_c': np.round(temperature_c, 2),
        'humidity_pct': np.round(humidity_pct, 2),
        'soil_moisture_pct': np.round(soil_moisture_pct, 2),
        'slope_deg': np.round(slope_deg, 2),
        'elevation_m': np.round(elevation_m, 2),
        'previous_rainfall_mm': np.round(previous_rainfall_mm, 2),
        'wind_speed_kmh': np.round(wind_speed_kmh, 2),
        'landslide_occurrence': landslide_occurrence
    })
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated landslide dataset with {len(df)} records saved to '{output_path}'.")
    print(f"Class distribution: No Landslide (0): {(df['landslide_occurrence']==0).sum()}, Landslide (1): {(df['landslide_occurrence']==1).sum()}")
    return df

if __name__ == "__main__":
    generate_landslide_dataset()
