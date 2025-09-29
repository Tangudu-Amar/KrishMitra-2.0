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
    'title': { 'en': 'KrishiMitra', 'hi': 'कृषिमित्रा', 'te': 'కృషిమిత్ర' },
    'dashboard': { 'en': 'Dashboard', 'hi': 'डैशबोर्ड', 'te': 'డాష్‌బోర్డ్' },
    'crop_recommendation': { 'en': 'Crop Recommendation', 'hi': 'फसल सिफारिश', 'te': 'పంట సిఫార్సు' },
    'enter_crop_details': { 'en': 'Enter Crop Details', 'hi': 'फसल का विवरण दर्ज करें', 'te': 'పంట వివరాలను నమోదు చేయండి' },
    'land_area': { 'en': 'Land Area (in acres):', 'hi': 'भूमि क्षेत्र (एकड़ में):', 'te': 'భూమి విస్తీర్ణం (ఎకరాలలో):' },
    'previous_crop': { 'en': 'Previous Crop:', 'hi': 'पिछली फसल:', 'te': 'గత పంట:' },
    'latitude': { 'en': 'Latitude:', 'hi': 'अक्षांश:', 'te': 'అక్షాంశం:' },
    'longitude': { 'en': 'Longitude:', 'hi': 'देशांतर:', 'te': 'రేఖాంశం:' },
    'state': { 'en': 'State:', 'hi': 'राज्य:', 'te': 'రాష్ట్రం:' },
    'district': { 'en': 'District:', 'hi': 'ज़िला:', 'te': 'జిల్లా:' },
    'use_my_gps': { 'en': 'Use My GPS Location', 'hi': 'मेरी जीपीएस लोकेशन का उपयोग करें', 'te': 'నా జీపీఎస్ స్థానాన్ని ఉపయోగించండి' },
    'get_recommendation': { 'en': 'Get Recommendation', 'hi': 'सिफारिश प्राप्त करें', 'te': 'సిఫార్సు పొందండి' },
    'model_recommendation': { 'en': 'Model Recommendation', 'hi': 'मॉडल सिफारिश', 'te': 'మోడల్ సిఫార్సు' },
    'results_display': { 'en': 'Results will be displayed here...', 'hi': 'परिणाम यहाँ प्रदर्शित होंगे...', 'te': 'ఫలితాలు ఇక్కడ చూపబడతాయి...' },
    'language': { 'en': 'Language:', 'hi': 'भाषा:', 'te': 'భాష:' },
    'fetching_recommendations': { 'en': 'Fetching recommendations...', 'hi': 'सिफारिशें प्राप्त हो रही हैं...', 'te': 'సిఫార్సులను పొందుతోంది...' },
    'paddy_rice': { 'en': 'Paddy (Rice)', 'hi': 'धान (चावल)', 'te': 'వరి (బియ్యం)' },
    'paddy_rice_subtext': { 'en': 'This crop is highly recommended based on your local soil and climate data.', 'hi': 'आपके स्थानीय मिट्टी और जलवायु डेटा के आधार पर इस फसल की अत्यधिक अनुशंसा की जाती है।', 'te': 'మీ స్థానిక నేల మరియు వాతావరణ డేటా ఆధారంగా ఈ పంట చాలా సిఫార్సు చేయబడింది.' },
    'sowing_season': { 'en': 'Sowing Season', 'hi': 'बुवाई का मौसम', 'te': 'విత్తే కాలం' },
    'kharif': { 'en': 'Kharif', 'hi': 'खरीफ', 'te': 'ఖరీఫ్' },
    'growth_duration': { 'en': 'Growth Duration', 'hi': 'बढ़ने की अवधि', 'te': 'పెరుగుదల వ్యవధి' },
    '120_days': { 'en': '120 Days', 'hi': '120 दिन', 'te': '120 రోజులు' },
    'expected_yield': { 'en': 'Expected Yield', 'hi': 'अपेक्षित उपज', 'te': 'ఆశించిన దిగుబడి' },
    'yield_value': { 'en': '4-5 tons/ha', 'hi': '4-5 टन/हेक्टेयर', 'te': '4-5 టన్నులు/హెక్టారు' },
    'new_recommendation': { 'en': 'Get New Recommendation', 'hi': 'नई सिफारिश प्राप्त करें', 'te': 'కొత్త సిఫార్సు పొందండి' },
    'market_prices': { 'en': 'Market Prices', 'hi': 'बाजार भाव', 'te': 'మార్కెట్ ధరలు' },
    'tomato': { 'en': 'Tomato', 'hi': 'टमाटर', 'te': 'టమాటో' },
    'onion': { 'en': 'Onion', 'hi': 'प्याज', 'te': 'ఉల్లిపాయ' },
    'potato': { 'en': 'Potato', 'hi': 'आलू', 'te': 'ఆలూ' },
    'cotton': { 'en': 'Cotton', 'hi': 'कपास', 'te': 'పత్తి' },
    'maize': { 'en': 'Maize', 'hi': 'मक्का', 'te': 'మొక్కజొన్న' },
    'wheat': { 'en': 'Wheat', 'hi': 'गेहूं', 'te': 'గోధుమ' },
    'district_required_error': { 'en': 'District information is required. Please use the GPS button or enter a district manually.', 'hi': 'जिले की जानकारी आवश्यक है। कृपया जीपीएस बटन का उपयोग करें या मैन्युअल रूप से एक जिला दर्ज करें।', 'te': 'జిల్లా సమాచారం తప్పనిసరి. దయచేసి GPS బటన్‌ని ఉపయోగించండి లేదా ఒక జిల్లాను మాన్యువల్‌గా నమోదు చేయండి.' },
    'recommended_crops': { 'en': 'Recommended Crops', 'hi': 'अनुशंसित फसलें', 'te': 'సిఫార్సు చేయబడిన పంటలు' },
    'detailed_report': { 'en': 'Detailed Report', 'hi': 'विस्तृत रिपोर्ट', 'te': 'వివరణాత్మక నివేదిక' },
    'report_unavailable': { 'en': 'Report not available. Please try again.', 'hi': 'रिपोर्ट उपलब्ध नहीं है। कृपया पुनः प्रयास करें।', 'te': 'నివేదిక అందుబాటులో లేదు. దయచేసి మళ్ళీ ప్రయత్నించండి.' }
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

def get_forecast(location: str, days: int = 10):
    """Fetches weather forecast for a given location."""
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
                "temperature": day["day"].get("avgtemp_c"),
                "humidity": day["day"].get("avghumidity"),
                "rainfall": day["day"].get("totalprecip_mm", 0)
            })
        return {"forecast": forecast, "success": True}
    except Exception as e:
        return {"error": f"Weather forecast failed: {str(e)}", "success": False}


# --- Routes ---
@app.route("/")
def home():
    if 'lang' not in session:
        session['lang'] = 'en'
    lang = session['lang']
    return render_template("index.html", lang=lang, translations=TRANSLATIONS)

@app.route("/crop-recommendation")
def crop_recommendation_page():
    if 'lang' not in session:
        session['lang'] = 'en'
    lang = session['lang']
    return render_template("text_input.html", lang=lang, translations=TRANSLATIONS)

@app.route("/change-language", methods=["POST"])
def change_language():
    lang = request.form.get("lang", "en")
    session['lang'] = lang
    return redirect(request.referrer or url_for('home'))


@app.route("/crop-prediction")
def crop_prediction_page():
    gemini_report = session.pop('gemini_report', None)
    crops = session.get("crops", [])
    if 'lang' not in session:
        session['lang'] = 'en'
    lang = session['lang']

    if not gemini_report and not crops:
        return redirect(url_for('crop_recommendation_page'))
    return render_template("result.html", gemini_report=gemini_report, crops=crops, lang=lang, translations=TRANSLATIONS)

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()
#     if not data or "district" not in data:
#         return jsonify({"error": "District not provided"}), 400

#     district = data["district"].strip().lower()
#     prev_crop = data.get("crop", "").strip().lower()
#     area = data.get("area", "").strip().lower()
#     lang = session.get('lang', 'en')

#     # --- 1. Soil Data ---
#     try:
#         district_data = df[df['District Name'].str.lower() == district]
#         if district_data.empty:
#             return jsonify({"error": f"Data for '{district.title()}' not available."}), 404
#         soil = district_data.iloc[0]
#         n, p, k, ph = soil['N'], soil['P'], soil['K'], soil['pH']
#     except KeyError:
#         return jsonify({"error": "Server error: Soil data format incorrect."}), 500

#     # --- 2. Weather Forecast (10-day avg) ---
#     forecast_data = get_forecast(district, days=10)
#     if not forecast_data["success"]:
#         return jsonify({"error": "Could not retrieve weather forecast."}), 500

#     forecast = forecast_data["forecast"]
#     avg_temperature = sum(d["temperature"] for d in forecast if d.get("temperature")) / len(forecast)
#     avg_humidity = sum(d["humidity"] for d in forecast if d.get("humidity")) / len(forecast)
#     avg_rainfall = sum(d["rainfall"] for d in forecast if d.get("rainfall")) / len(forecast)

#     # --- 3. Predict Crops ---
#     input_features = pd.DataFrame(
#         [[n, p, k, avg_temperature, avg_humidity, ph, avg_rainfall]],
#         columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
#     )
#     probabilities = model.predict_proba(input_features)[0]
#     top_indices = np.argsort(probabilities)[-3:][::-1]
#     recommendations = [CROP_LABELS[i] for i in top_indices]

#     # --- 4. Gemini Report ---
#     prompt = f"""
#     You are an expert AI agronomist named KrishiMitra. Your task is to provide a detailed, well-structured crop recommendation in HTML format based on the data provided below. Your response must be only the HTML code, ready to be rendered directly on a webpage. Do not include <html> or <body> tags.

#     ## Farm & Environmental Data:
#     - Soil Nutrients: Nitrogen (N): {n} ppm, Phosphorus (P): {p} ppm, Potassium (K): {k} ppm
#     - Soil pH: {ph}
#     - 10-Day Weather Forecast:
#       - Average Temperature: {avg_temperature:.2f} °C
#       - Average Humidity: {avg_humidity:.2f}%
#       - Total Rainfall: {avg_rainfall:.2f} mm
#     - Farm Details:
#       - Previous Crop Harvested: {prev_crop if prev_crop else 'N/A'}
#       - Land Area: {area if area else 'N/A'} acres
#     - Pre-selected Viable Crops (from analysis): {', '.join(recommendations)}

#     ## Instructions:
#     From the list of pre-selected crops, choose the single best crop as the primary recommendation. Provide a detailed justification, research basis, and suggest 2-3 other feasible alternatives from the list. Format your entire response using the HTML structure provided below. Do not add any text outside of the main <div class="report-container"> wrapper. Respond entirely in the language corresponding to the language code: {lang}.

#     HTML STRUCTURE:
#     <div class="report-container" style="font-family: Arial, sans-serif; color: #333;">
#         <div class="card primary-recommendation" style="background-color: #e8f5e9; border-left: 5px solid #4CAF50; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
#             <h2 style="margin-top: 0;">Primary Recommendation: [Name of the Best Crop]</h2>
#             <p>Based on a comprehensive analysis of your soil and local weather, <strong>[Crop Name]</strong> is the most profitable and suitable crop for the upcoming season.</p>
#         </div>
#         <div class="card details" style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
#             <h3>Detailed Analysis</h3><hr>
#             <h4>Justification for Recommendation</h4>
#             <ul>
#                 <li><strong>Soil Compatibility:</strong> Your soil's NPK values and pH are within the ideal range required for robust growth of [Crop Name].</li>
#                 <li><strong>Climate Suitability:</strong> The current weather forecast aligns perfectly with the critical germination and growing phases for this crop.</li>
#                 <li><strong>Crop Rotation Benefit:</strong> Following a previous crop of [{prev_crop if prev_crop else 'N/A'}], planting [Crop Name] can help improve soil health by [mention a specific benefit, e.g., breaking pest cycles, restoring nitrogen, etc.].</li>
#             </ul>
#             <h4>Research Basis</h4>
#             <ul>
#                 <li>Agricultural studies for your geo-climatic zone show that [Crop Name] has a high yield potential under these specific soil and weather parameters.</li>
#                 <li>This crop is recommended for this season due to its [mention a key trait, e.g., drought tolerance, water-use efficiency, etc.], which is supported by regional agricultural university research.</li>
#             </ul>
#         </div>
#         <div class="card alternatives" style="background-color: #e3f2fd; padding: 15px; border-radius: 5px;">
#             <h3>Feasible Alternative Crops</h3><hr>
#             <p><strong>1. [Alternative Crop 1 Name]:</strong> A strong secondary option. It thrives in similar conditions but may have a different market value or water requirement.</p>
#             <p><strong>2. [Alternative Crop 2 Name]:</strong> Consider this crop if you are looking for a shorter harvest cycle or better resistance to local pests.</p>
#         </div>
#     </div>
#     """

#     try:
#         response = gemini_model.generate_content(prompt)
#         gemini_report = response.text
#     except Exception as e:
#         print(f"Gemini API call failed: {e}")
#         gemini_report = f"<p>Error generating report: {e}</p>"

#     # --- 5. Save and Return ---
#     session["crops"] = recommendations
#     session["gemini_report"] = gemini_report

#     return jsonify({
#         "status": "success",
#         "redirect_url": url_for('crop_prediction_page')
#     })

@app.route("/predict", methods=["POST"])
def predict():
    lang = session.get('lang', 'en')
    data = request.form
    if not data or "district" not in data:
        return "District not provided", 400

    district = data.get("district", "").strip().lower()
    prev_crop = data.get("crop", "").strip().lower()
    area = data.get("area", "").strip().lower()


    if not district:
        return jsonify({"error": "District not provided"}), 400

    # --- Soil Data ---
    try:
        district_data = df[df['District Name'].str.lower() == district]
        if district_data.empty:
            return jsonify({"error": f"Data for '{district.title()}' not available."}), 404
        soil = district_data.iloc[0]
        n, p, k, ph = soil['N'], soil['P'], soil['K'], soil['pH']
    except KeyError:
        return jsonify({"error": "Server error: Soil data format incorrect."}), 500

    # --- Weather Forecast ---
    forecast_data = get_forecast(district, days=10)
    if not forecast_data["success"]:
        return jsonify({"error": "Could not retrieve weather forecast."}), 500

    forecast = forecast_data["forecast"]
    avg_temperature = sum(d["temperature"] for d in forecast if d.get("temperature")) / len(forecast)
    avg_humidity = sum(d["humidity"] for d in forecast if d.get("humidity")) / len(forecast)
    avg_rainfall = sum(d["rainfall"] for d in forecast if d.get("rainfall")) / len(forecast)

    # --- Crop Prediction ---
    user_data = pd.DataFrame(
            [[n, p, k, avg_temperature, avg_humidity, ph, avg_rainfall]],
            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            )

    # --- Predict probabilities ---
    probabilities_stacked = model.predict_proba(user_data)  # list of arrays
    probabilities = np.array(probabilities_stacked).T  # shape: (n_samples, n_labels)

    # --- Crop labels from y_train ---
    crop_labels = CROP_LABELS  # ensure this matches training labels

    # --- Apply threshold ---
    threshold = 0.5
    recommended_crops_indices = np.where(probabilities[0] > threshold)[0]

    if len(recommended_crops_indices) > 0:
        recommendations = [crop_labels[i] for i in recommended_crops_indices]
    else:
        # fallback to crop with highest probability
        highest_prob_index = np.argmax(probabilities[0])
        recommendations = [crop_labels[highest_prob_index]]

    # --- Gemini report generation remains unchanged ---

    print(",".join(recommendations))
    prompt = f"""
        You are an expert AI agronomist named KrishiMitra. Your task is to provide a detailed, well-structured crop recommendation in HTML format based on the data provided below. Your response must be only the HTML code, ready to be rendered directly on a webpage. Do not include <html> or <body> tags.

        ## Farm & Environmental Data:
        - Soil Nutrients: Nitrogen (N): {n} ppm, Phosphorus (P): {p} ppm, Potassium (K): {k} ppm
        - Soil pH: {ph}
        - 10-Day Weather Forecast:
        - Average Temperature: {avg_temperature:.2f} °C
        - Average Humidity: {avg_humidity:.2f}%
        - Total Rainfall: {avg_rainfall:.2f} mm
        - Farm Details:
        - Previous Crop Harvested: {prev_crop if prev_crop else 'N/A'}
        - Land Area: {area if area else 'N/A'} acres
        - Pre-selected Viable Crops (from analysis): {', '.join(recommendations)}

        ## Instructions:
        From the list of pre-selected crops, choose the single best crop as the primary recommendation. Provide a detailed justification, research basis, and suggest 2-3 other feasible alternatives from the list. Format your entire response using the HTML structure provided below. Do not add any text outside of the main <div class="report-container"> wrapper. Respond entirely in the language corresponding to the language code: {lang}.

        HTML STRUCTURE:
        <div class="report-container" style="font-family: Arial, sans-serif; color: #333;">
            <div class="card primary-recommendation" style="background-color: #e8f5e9; border-left: 5px solid #4CAF50; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <h2 style="margin-top: 0;">Primary Recommendation: [Name of the Best Crop]</h2>
                <p>Based on a comprehensive analysis of your soil and local weather, <strong>[Crop Name]</strong> is the most profitable and suitable crop for the upcoming season.</p>
            </div>
            <div class="card details" style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <h3>Detailed Analysis</h3><hr>
                <h4>Justification for Recommendation</h4>
                <ul>
                    <li><strong>Soil Compatibility:</strong> Your soil's NPK values and pH are within the ideal range required for robust growth of [Crop Name].</li>
                    <li><strong>Climate Suitability:</strong> The current weather forecast aligns perfectly with the critical germination and growing phases for this crop.</li>
                    <li><strong>Crop Rotation Benefit:</strong> Following a previous crop of [{prev_crop if prev_crop else 'N/A'}], planting [Crop Name] can help improve soil health by [mention a specific benefit, e.g., breaking pest cycles, restoring nitrogen, etc.].</li>
                </ul>
                <h4>Research Basis</h4>
                <ul>
                    <li>Agricultural studies for your geo-climatic zone show that [Crop Name] has a high yield potential under these specific soil and weather parameters.</li>
                    <li>This crop is recommended for this season due to its [mention a key trait, e.g., drought tolerance, water-use efficiency, etc.], which is supported by regional agricultural university research.</li>
                </ul>
            </div>
            <div class="card alternatives" style="background-color: #e3f2fd; padding: 15px; border-radius: 5px;">
                <h3>Feasible Alternative Crops</h3><hr>
                <p><strong>1. [Alternative Crop 1 Name]:</strong> A strong secondary option. It thrives in similar conditions but may have a different market value or water requirement.</p>
                <p><strong>2. [Alternative Crop 2 Name]:</strong> Consider this crop if you are looking for a shorter harvest cycle or better resistance to local pests.</p>
            </div>
        </div>
        """
    try:
        response = gemini_model.generate_content(prompt)
        gemini_report = response.text
    except Exception as e:
        print(f"Gemini API call failed: {e}")
        gemini_report = f"<p>Error generating report: {e}</p>"

    session["crops"] = recommendations
    session["gemini_report"] = gemini_report

    # If form submission, redirect to results page
    if not request.is_json:
        return redirect(url_for('crop_prediction_page'))

    # If JSON request, return JSON response
    return redirect(url_for('crop_prediction_page'))


if __name__ == "__main__":
    app.run(debug=True)