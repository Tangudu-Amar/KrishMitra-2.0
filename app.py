from flask import Flask, request, jsonify, render_template, url_for, session, redirect
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import requests
from dotenv import load_dotenv
import google.generativeai as genai
from model_definitions import RFXGEnsemble

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = "http://api.weatherapi.com/v1"

CROP_LABELS = [
    'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton',
    'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans',
    'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate',
    'rice', 'watermelon'
]

# --- App Initialization ---
app = Flask(__name__)
app.secret_key = os.urandom(24) 
CORS(app)

# --- Configure Gemini API ---
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('models/gemini-2.0-flash')

# --- Load Data and Model ---
try:
    model = joblib.load("crop_model.pkl")
    df = pd.read_excel("Model_Data.xlsx") 
except FileNotFoundError as e:
    print(f"Error loading data file or model: {e}")
    exit()

# --- Helper Functions ---
def get_current_weather(location: str):
    """Fetches current weather data for a given location."""
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
    except requests.exceptions.RequestException as e:
        print(f"Weather API request error: {e}")
        return {"error": str(e), "success": False}

# --- Flask Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/crop-recommendation")
def crop_recommendation_page():
    return render_template("text_input.html")

@app.route("/crop-prediction")
def crop_prediction_page():
    gemini_report = session.pop('gemini_report', None)
    crops = session.get("crops", [])

    if not gemini_report and not crops:
        return redirect(url_for('crop_recommendation_page'))
    return render_template("result.html", gemini_report=gemini_report, crops=crops)

def get_forecast(location: str, days: int = 10):
    """
    Fetch weather forecast (default: 10 days) for a given location.
    """
    try:
        url = f"{BASE_URL}/forecast.json"
        params = {"key": WEATHER_API_KEY, "q": location, "days": days}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        forecast = []
        for day in data["forecast"]["forecastday"]:
            forecast.append({
                "date": day["date"],
                "temperature": day["day"].get("avgtemp_c"),  # Avg temperature for the day
                "humidity": day["day"].get("avghumidity"),  # Avg humidity for the day
                "rainfall": day["day"].get("totalprecip_mm", 0)  # Total rainfall for the day
            })

        return {
            "forecast": forecast,
            "success": True
        }
    except requests.exceptions.RequestException as e:
        return {
            "error": f"Weather forecast request failed: {str(e)}",
            "success": False
        }
    except KeyError as e:
        return {
            "error": f"Weather forecast parsing failed: {str(e)}",
            "success": False
        }


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "district" not in data:
        return jsonify({"error": "District not provided"}), 400

    district = data["district"].strip().lower()
    prev_crop = data.get("crop", "").strip().lower()
    area = data.get("area", "").strip().lower()

    # 1. Get Soil Data
    try:
        district_data = df[df['District Name'].str.lower() == district]
        if district_data.empty:
            return jsonify({"error": f"Data for '{district.title()}' not available."}), 404
        soil = district_data.iloc[0]
        n, p, k, ph = soil['N'], soil['P'], soil['K'], soil['pH']
    except KeyError:
        return jsonify({"error": "Server error: Soil data format is incorrect."}), 500

    # 2. Get Weather Forecast (10 days avg)
    forecast_data = get_forecast(district, days=10)
    if not forecast_data["success"]:
        return jsonify({"error": "Could not retrieve weather forecast."}), 500

    forecast = forecast_data["forecast"]
    avg_temperature = sum(d["temperature"] for d in forecast if d["temperature"]) / len(forecast)
    avg_humidity = sum(d["humidity"] for d in forecast if d["humidity"]) / len(forecast)
    avg_rainfall = sum(d["rainfall"] for d in forecast if d["rainfall"]) / len(forecast)

    # 3. Predict Top Crops using avg values
    input_features = pd.DataFrame([[n, p, k, avg_temperature, avg_humidity, ph, avg_rainfall]],
                                  columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    probabilities = model.predict_proba(input_features)[0]
    top_indices = np.argsort(probabilities)[-3:][::-1]
    recommendations = [CROP_LABELS[i] for i in top_indices]

    # 4. Generate Detailed Report with Gemini API
    prompt = f"""
    You are KrishiMitra, an agricultural assistant. Based on the following factors:
    - Nitrogen: {n}, Phosphorus: {p}, Potassium: {k}
    - Soil pH: {ph}
    - **10-Day Average Weather**:
        - Temperature: {avg_temperature:.2f} °C
        - Humidity: {avg_humidity:.2f}%
        - Rainfall: {avg_rainfall:.2f} mm
    - Previous Crop: {prev_crop if prev_crop else 'N/A'}
    - Land Area: {area if area else 'N/A'}

    Recommend suitable crops from: {', '.join(recommendations)}.
    Provide explanation in structured format:
    - Recommended Crops
    - Why these crops
    - Ideal soil conditions
    - Crop suggestion based on season
    Summarize briefly with small points for each heading.
    (give in the form of html to render directly on frontend dont give html in start and end backticks)
    """

    try:
        response = gemini_model.generate_content(prompt)
        gemini_report = response.text
    except Exception as e:
        print(f"Gemini API call failed: {e}")
        gemini_report = None

    # 5. Save to session
    session["crops"] = recommendations
    session["gemini_report"] = gemini_report

    print("DEBUG: Saved to session", session["crops"], session["gemini_report"])

    return jsonify({
        "status": "success",
        "redirect_url": url_for('crop_prediction_page')
    })


if __name__ == "__main__":
    app.run(debug=True)
