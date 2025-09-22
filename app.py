from flask import Flask, request, jsonify,render_template
import joblib
import pandas as pd
import numpy as np
import os
import requests
from dotenv import load_dotenv
from model_definitions import RFXGEnsemble

load_dotenv()

# Weather API config
WEATHER_API_KEY = "19ea267632424337a86101830252209"
BASE_URL = "http://api.weatherapi.com/v1"

# Load model
model = joblib.load("crop_model.pkl")

# Crop labels
CROP_LABELS = [
    'apple','banana','blackgram','chickpea','coconut','coffee','cotton',
    'grapes','jute','kidneybeans','lentil','maize','mango','mothbeans',
    'mungbean','muskmelon','orange','papaya','pigeonpeas','pomegranate',
    'rice','watermelon'
]

# Load district soil data
df = pd.read_excel("Model_Data.xlsx")  # must contain 'district', 'N', 'P', 'K', 'ph'

app = Flask(__name__)

# Helper function to fetch current weather
def get_current_weather(location: str):
    try:
        url = f"{BASE_URL}/current.json"
        params = {"key": WEATHER_API_KEY, "q": location}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return {
            "temperature": data["current"].get("temp_c"),
            "humidity": data["current"].get("humidity"),
            "rainfall": data["current"].get("precip_mm", 0),
            "success": True
        }
    except Exception as e:
        return {"error": str(e), "success": False}

@app.route("/")
def home():
    return render_template("index.html")

# 2️⃣ Crop recommendation input page
@app.route("/crop-recommendation")
def crop_recommendation_page():
    return render_template("text_input.html")  # Make sure this file exists in templates folder


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "district" not in data:
        return jsonify({"error": "Please provide a district"}), 400

    district = data["district"].strip().lower()

    # Get district soil data
    district_data = df[df['District Name'].str.lower() == district]
    if district_data.empty:
        return jsonify({"error": f"District '{district}' not found"}), 404

    soil = district_data.iloc[0]
    n, p, k, ph = soil['N'], soil['P'], soil['K'], soil['pH']

    # Get weather data
    weather = get_current_weather(district)
    if not weather["success"]:
        return jsonify({"error": f"Weather API failed: {weather.get('error', '')}"}), 500

    temperature = weather['temperature']
    humidity = weather['humidity']
    rainfall = weather['rainfall']

    # Prepare dataframe for model
    user_data = pd.DataFrame([[n, p, k, temperature, humidity, ph, rainfall]],
                             columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

    # Predict
    probabilities_stacked = model.predict_proba(user_data)
    probabilities = probabilities_stacked.T
    recommended_crops_indices = np.where(probabilities[0] > 0.5)[0]

    if len(recommended_crops_indices) > 0:
        recommendations = [CROP_LABELS[i] for i in recommended_crops_indices]
    else:
        best_idx = np.argmax(probabilities[0])
        recommendations = [f"{CROP_LABELS[best_idx]} (prob {probabilities[0,best_idx]:.2f})"]

    return jsonify({
        "status": "success",
        "district": district,
        "input_features": {
            "N": n, "P": p, "K": k, "ph": ph,
            "temperature_C": temperature,
            "humidity_percent": humidity,
            "rainfall_mm_last_hr": rainfall
        },
        "recommended_crops": [{"crop": c, "probability": float(probabilities[0][i]) if len(recommended_crops_indices) > 0 else 1.0}
                              for i, c in enumerate(recommendations)]
    })



if __name__ == "__main__":
    app.run(debug=True)
