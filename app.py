import os
import sys
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Append project root
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.predictor import LandslidePredictor

# Page Configuration
st.set_page_config(
    page_title="Landslide Risk Predictor",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean CSS Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Make input label text (Rainfall, Moisture, etc.) bright white */
    label, 
    label p,
    label div,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    .stNumberInput label,
    .stNumberInput label p {
        color: #ffffff !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    .stMarkdown h5 {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }
    
    .header-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
    }
    .header-box h2 {
        color: #38bdf8;
        margin-bottom: 6px;
        font-weight: 700;
    }
    .header-box p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
    }
    
    .sensor-info-card {
        background: #1e293b;
        border-left: 4px solid #38bdf8;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 20px;
        font-size: 0.9rem;
        color: #cbd5e1;
    }

    .result-container {
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    
    .res-high {
        background: #1e3a8a;
        border: 2px solid #3b82f6;
        color: #dbeafe;
    }
    .res-medium {
        background: #1e293b;
        border: 2px solid #0284c7;
        color: #93c5fd;
    }
    .res-low {
        background: #0f172a;
        border: 2px solid #2563eb;
        color: #bfdbfe;
    }
    
    /* Left sidebar presets and radio controls - styled in blue */
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        color: #38bdf8 !important;
        font-weight: 600 !important;
    }
    
    .prob-display {
        font-size: 3.5rem;
        font-weight: 900;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load Random Forest Predictor
@st.cache_resource
def get_predictor():
    return LandslidePredictor(models_dir="models")

predictor = get_predictor()

# Clean Header
st.markdown("""
<div class="header-box">
    <h2>⛰️ Landslide Prediction & Early Warning System</h2>
    <p>Software prototype for landslide risk assessment using Random Forest Machine Learning model.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sensor-info-card">
    <strong>💡 Hardware Sensor Context:</strong> In a deployed IoT system, environmental parameters (Rainfall, Soil Moisture, Temperature, Humidity, Slope) are collected automatically via hardware sensors (such as Rain Gauge, Soil Probe, DHT22, MPU6050 accelerometer) connected to an ESP32 microcontroller. Here, you can manually input or test values to view immediate predictions.
</div>
""", unsafe_allow_html=True)

# Sidebar Preset Controls
st.sidebar.title("🎯 Quick Presets")
preset = st.sidebar.radio(
    "Select Scenario Preset:",
    ["🚨 High Risk (Heavy Rain & Steep Slope)", "⚠️ Medium Risk (Moderate Rain)", "✅ Low Risk (Fair Weather)", "Custom Manual Entry"],
    index=0
)

# Preset Values
if preset == "🚨 High Risk (Heavy Rain & Steep Slope)":
    init = {'rain': 180.0, 'prev_rain': 120.0, 'moisture': 74.0, 'slope': 38.0, 'temp': 19.0, 'hum': 88.0, 'elev': 950.0, 'wind': 40.0}
elif preset == "⚠️ Medium Risk (Moderate Rain)":
    init = {'rain': 65.0, 'prev_rain': 45.0, 'moisture': 52.0, 'slope': 24.0, 'temp': 22.0, 'hum': 70.0, 'elev': 600.0, 'wind': 20.0}
elif preset == "✅ Low Risk (Fair Weather)":
    init = {'rain': 10.0, 'prev_rain': 5.0, 'moisture': 20.0, 'slope': 12.0, 'temp': 26.0, 'hum': 45.0, 'elev': 300.0, 'wind': 10.0}
else:
    init = {'rain': 100.0, 'prev_rain': 60.0, 'moisture': 55.0, 'slope': 28.0, 'temp': 21.0, 'hum': 75.0, 'elev': 700.0, 'wind': 25.0}

col_in, col_res = st.columns([1.1, 1.0])

with col_in:
    st.subheader("📥 Enter Environmental Parameters")
    
    st.markdown("##### 🌧️ Precipitation & Moisture (Rain Gauge & Soil Probe)")
    c1, c2 = st.columns(2)
    with c1:
        rainfall = st.number_input("24h Rainfall (mm)", min_value=0.0, max_value=400.0, value=float(init['rain']), step=5.0)
        soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=float(init['moisture']), step=1.0)
    with c2:
        prev_rainfall = st.number_input("3-Day Antecedent Rain (mm)", min_value=0.0, max_value=350.0, value=float(init['prev_rain']), step=5.0)
        humidity = st.number_input("Humidity (%) [DHT22]", min_value=0.0, max_value=100.0, value=float(init['hum']), step=1.0)

    st.markdown("##### 📐 Terrain & Climate (Accelerometer & Weather Sensors)")
    c3, c4 = st.columns(2)
    with c3:
        slope = st.number_input("Slope Angle (°)", min_value=0.0, max_value=60.0, value=float(init['slope']), step=1.0)
        temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=float(init['temp']), step=0.5)
    with c4:
        elevation = st.number_input("Elevation (m)", min_value=0.0, max_value=4000.0, value=float(init['elev']), step=50.0)
        wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=120.0, value=float(init['wind']), step=1.0)

    input_payload = {
        'rainfall_mm': rainfall,
        'temperature_c': temperature,
        'humidity_pct': humidity,
        'soil_moisture_pct': soil_moisture,
        'slope_deg': slope,
        'elevation_m': elevation,
        'previous_rainfall_mm': prev_rainfall,
        'wind_speed_kmh': wind_speed
    }

with col_res:
    st.subheader("🎯 Prediction Result (Random Forest Model)")
    
    # Predict using Random Forest
    res = predictor.predict(input_payload, model_name="Random Forest")
    prob_pct = res['probability_pct']
    risk_level = res['risk_level']
    risk_code = res['risk_code']
    badge_color = res['badge_color']
    
    card_css = "res-high" if risk_code == "HIGH" else ("res-medium" if risk_code == "MEDIUM" else "res-low")
    
    st.markdown(f"""
    <div class="result-container {card_css}">
        <div style="font-size: 1.1rem; font-weight: 700; text-transform: uppercase;">
            Risk Category: {risk_level}
        </div>
        <div class="prob-display">{prob_pct}%</div>
        <div style="font-size: 0.95rem;">Landslide Probability</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress Bar Gauge
    st.markdown("#### Risk Gauge")
    st.progress(min(1.0, prob_pct / 100.0))
    
    # Advisory Box
    if risk_code == "HIGH":
        st.error(f"🚨 **HIGH RISK ALERT**: {res['advice']}")
    elif risk_code == "MEDIUM":
        st.warning(f"⚠️ **MODERATE RISK ADVISORY**: {res['advice']}")
    else:
        st.success(f"✅ **LOW RISK (SAFE)**: {res['advice']}")
        
    # Top Contributing Factors
    st.markdown("#### 🔑 Main Contributing Factors")
    factors = res['contributing_factors'][:3]
    for f in factors:
        st.write(f"- **{f['label']}**: {f['val']} (Impact: {f['impact_pct']}%)")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem;">
    Random Forest Landslide Early Warning System Prototype | Simplified User Interface
</div>
""", unsafe_allow_html=True)
