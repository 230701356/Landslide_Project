import time
import random
import numpy as np

class ESP32SensorSimulator:
    def __init__(self, node_id="ESP32_GEO_STATION_01", location="Station Alpha (Steep Slope Zone B)"):
        self.node_id = node_id
        self.location = location
        self.packet_counter = 0

    def generate_telemetry_frame(self, scenario="Monsoon Storm Alert"):
        """
        Generates simulated real-time telemetry packet from an ESP32 micro-controller board
        with attached weather and soil sensors.
        """
        self.packet_counter += 1
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        if scenario == "Normal Weather / Fair Season":
            rainfall = random.uniform(0.0, 15.0)
            prev_rainfall = random.uniform(5.0, 30.0)
            soil_moisture = random.uniform(20.0, 45.0)
            slope = random.uniform(15.0, 25.0)
            temp = random.uniform(20.0, 30.0)
            humidity = random.uniform(50.0, 70.0)
            wind = random.uniform(5.0, 20.0)
            elevation = 650.0

        elif scenario == "Heavy Monsoon Storm Alert":
            rainfall = random.uniform(130.0, 240.0)
            prev_rainfall = random.uniform(100.0, 210.0)
            soil_moisture = random.uniform(70.0, 95.0)
            slope = random.uniform(32.0, 48.0)
            temp = random.uniform(18.0, 24.0)
            humidity = random.uniform(85.0, 98.0)
            wind = random.uniform(35.0, 65.0)
            elevation = 650.0

        elif scenario == "Steep Slope Saturation":
            rainfall = random.uniform(90.0, 160.0)
            prev_rainfall = random.uniform(70.0, 150.0)
            soil_moisture = random.uniform(65.0, 88.0)
            slope = random.uniform(40.0, 55.0)
            temp = random.uniform(16.0, 22.0)
            humidity = random.uniform(80.0, 95.0)
            wind = random.uniform(15.0, 35.0)
            elevation = 1200.0

        else: # Dynamic fluctuating scenario
            rainfall = random.uniform(10.0, 200.0)
            prev_rainfall = random.uniform(10.0, 180.0)
            soil_moisture = random.uniform(25.0, 90.0)
            slope = random.uniform(10.0, 50.0)
            temp = random.uniform(15.0, 32.0)
            humidity = random.uniform(40.0, 98.0)
            wind = random.uniform(5.0, 50.0)
            elevation = 850.0

        telemetry = {
            "device_metadata": {
                "hardware": "ESP32-WROOM-32",
                "node_id": self.node_id,
                "location": self.location,
                "packet_seq": self.packet_counter,
                "timestamp": timestamp,
                "firmware_ver": "v2.4.1-IoT-ML"
            },
            "sensor_readings": {
                "rainfall_mm": round(float(rainfall), 2),
                "temperature_c": round(float(temp), 2),
                "humidity_pct": round(float(humidity), 2),
                "soil_moisture_pct": round(float(soil_moisture), 2),
                "slope_deg": round(float(slope), 2),
                "elevation_m": round(float(elevation), 2),
                "previous_rainfall_mm": round(float(prev_rainfall), 2),
                "wind_speed_kmh": round(float(wind), 2)
            }
        }
        return telemetry
