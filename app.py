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

# --- Translation Data ---
TRANSLATIONS = {
    'title': {
        'en': 'KrishiMitra',
        'hi': 'कृषिमित्रा',
        'te': 'కృషిమిత్ర'
    },
    'dashboard': {
        'en': 'Dashboard',
        'hi': 'डैशबोर्ड',
        'te': 'డాష్‌బోర్డ్'
    },
    'crop_recommendation': {
        'en': 'Crop Recommendation',
        'hi': 'फसल सिफारिश',
        'te': 'పంట సిఫార్సు'
    },
    'enter_crop_details': {
        'en': 'Enter Crop Details',
        'hi': 'फसल का विवरण दर्ज करें',
        'te': 'పంట వివరాలను నమోదు చేయండి'
    },
    'land_area': {
        'en': 'Land Area (in acres):',
        'hi': 'भूमि क्षेत्र (एकड़ में):',
        'te': 'భూమి విస్తీర్ణం (ఎకరాలలో):'
    },
    'previous_crop': {
        'en': 'Previous Crop:',
        'hi': 'पिछली फसल:',
        'te': 'గత పంట:'
    },
    'latitude': {
        'en': 'Latitude:',
        'hi': 'अक्षांश:',
        'te': 'అక్షాంశం:'
    },
    'longitude': {
        'en': 'Longitude:',
        'hi': 'देशांतर:',
        'te': 'రేఖాంశం:'
    },
    'state': {
        'en': 'State:',
        'hi': 'राज्य:',
        'te': 'రాష్ట్రం:'
    },
    'district': {
        'en': 'District:',
        'hi': 'ज़िला:',
        'te': 'జిల్లా:'
    },
    'use_my_gps': {
        'en': 'Use My GPS Location',
        'hi': 'मेरी जीपीएस लोकेशन का उपयोग करें',
        'te': 'నా జీపీఎస్ స్థానాన్ని ఉపయోగించండి'
    },
    'get_recommendation': {
        'en': 'Get Recommendation',
        'hi': 'सिफारिश प्राप्त करें',
        'te': 'సిఫార్సు పొందండి'
    },
    'model_recommendation': {
        'en': 'Model Recommendation',
        'hi': 'मॉडल सिफारिश',
        'te': 'మోడల్ సిఫార్సు'
    },
    'results_display': {
        'en': 'Results will be displayed here...',
        'hi': 'परिणाम यहाँ प्रदर्शित होंगे...',
        'te': 'ఫలితాలు ఇక్కడ చూపబడతాయి...'
    },
    'language': {
        'en': 'Language:',
        'hi': 'भाषा:',
        'te': 'భాష:'
    },
    'fetching_recommendations': {
        'en': 'Fetching recommendations...',
        'hi': 'सिफारिशें प्राप्त हो रही हैं...',
        'te': 'సిఫార్సులను పొందుతోంది...'
    },
    'paddy_rice': {
        'en': 'Paddy (Rice)',
        'hi': 'धान (चावल)',
        'te': 'వరి (బియ్యం)'
    },
    'paddy_rice_subtext': {
        'en': 'This crop is highly recommended based on your local soil and climate data.',
        'hi': 'आपके स्थानीय मिट्टी और जलवायु डेटा के आधार पर इस फसल की अत्यधिक अनुशंसा की जाती है।',
        'te': 'మీ స్థానిక నేల మరియు వాతావరణ డేటా ఆధారంగా ఈ పంట చాలా సిఫార్సు చేయబడింది.'
    },
    'sowing_season': {
        'en': 'Sowing Season',
        'hi': 'बुवाई का मौसम',
        'te': 'విత్తే కాలం'
    },
    'kharif': {
        'en': 'Kharif',
        'hi': 'खरीफ',
        'te': 'ఖరీఫ్'
    },
    'growth_duration': {
        'en': 'Growth Duration',
        'hi': 'बढ़ने की अवधि',
        'te': 'పెరుగుదల వ్యవధి'
    },
    '120_days': {
        'en': '120 Days',
        'hi': '120 दिन',
        'te': '120 రోజులు'
    },
    'expected_yield': {
        'en': 'Expected Yield',
        'hi': 'अपेक्षित उपज',
        'te': 'ఆశించిన దిగుబడి'
    },
    'yield_value': {
        'en': '4-5 tons/ha',
        'hi': '4-5 टन/हेक्टेयर',
        'te': '4-5 టన్నులు/హెక్టారు'
    },
    'new_recommendation': {
        'en': 'Get New Recommendation',
        'hi': 'नई सिफारिश प्राप्त करें',
        'te': 'కొత్త సిఫార్సు పొందండి'
    },
    'market_prices': {
        'en': 'Market Prices',
        'hi': 'बाजार भाव',
        'te': 'మార్కెట్ ధరలు'
    },
    'tomato': {
        'en': 'Tomato',
        'hi': 'टमाटर',
        'te': 'టమాటో'
    },
    'onion': {
        'en': 'Onion',
        'hi': 'प्याज',
        'te': 'ఉల్లిపాయ'
    },
    'potato': {
        'en': 'Potato',
        'hi': 'आलू',
        'te': 'ఆలూ'
    },
    'cotton': {
        'en': 'Cotton',
        'hi': 'कपास',
        'te': 'పత్తి'
    },
    'maize': {
        'en': 'Maize',
        'hi': 'मक्का',
        'te': 'మొక్కజొన్న'
    },
    'wheat': {
        'en': 'Wheat',
        'hi': 'गेहूं',
        'te': 'గోధుమ'
    },
    'district_required_error': {
        'en': 'District information is required. Please use the GPS button or enter a district manually.',
        'hi': 'जिले की जानकारी आवश्यक है। कृपया जीपीएस बटन का उपयोग करें या मैन्युअल रूप से एक जिला दर्ज करें।',
        'te': 'జిల్లా సమాచారం తప్పనిసరి. దయచేసి GPS బటన్‌ని ఉపయోగించండి లేదా ఒక జిల్లాను మాన్యువల్‌గా నమోదు చేయండి.'
    },
    'recommended_crops': {
        'en': 'Recommended Crops',
        'hi': 'अनुशंसిత పంటలు',
        'te': 'సిఫార్సు చేయబడిన పంటలు'
    },
    'detailed_report': {
        'en': 'Detailed Report',
        'hi': 'వివరణాత్మక నివేదిక',
        'te': 'వివరణాత్మక నివేదిక'
    },
    'report_unavailable': {
        'en': 'Report not available. Please try again.',
        'hi': 'रिपोर्ट उपलब्ध नहीं है। कृपया पुनः प्रयास करें।',
        'te': 'నివేదిక అందుబాటులో లేదు. దయచేసి మళ్ళీ ప్రయత్నించండి.'
    }
}


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
    if 'lang' not in session:
        session['lang'] = 'en'
    lang = session['lang']
    return render_template("index.html", translations=TRANSLATIONS, lang=lang)

@app.route("/crop-recommendation")
def crop_recommendation_page():
    if 'lang' not in session:
        session['lang'] = 'en'
    lang = session['lang']
    return render_template("text_input.html", translations=TRANSLATIONS, lang=lang)

@app.route("/change-language", methods=["POST"])
def change_language():
    data = request.get_json()
    lang = data.get("lang", "en")
    session['lang'] = lang
    return jsonify({"status": "success", "lang": lang})

@app.route("/crop-prediction")
def crop_prediction_page():
    gemini_report = session.pop('gemini_report', None)
    crops = session.get("crops", [])

    if 'lang' not in session:
        session['lang'] = 'en'
    lang = session['lang']
    
    if not gemini_report and not crops:
        return redirect(url_for('crop_recommendation_page'))
    return render_template("result.html", gemini_report=gemini_report, crops=crops, translations=TRANSLATIONS, lang=lang)

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
    lang = session.get('lang', 'en')

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
    - give for first recomendation only.
    - why that crop.
    - support the recomendation provided.
    - dont unnecessarily send input data back in reply.
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
