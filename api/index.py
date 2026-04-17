import os

from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

crop_pred_model = joblib.load(os.path.join(BASE_DIR, "../crop_model.joblib"))
fert_pred_model = joblib.load(os.path.join(BASE_DIR, "../fertilizer_model.joblib"))
crop_advice = {
    "rice": {
        "temp": "20°C to 35°C",
        "tips": [
            "Requires flooded or waterlogged conditions",
            "High nitrogen demand during vegetative stage",
            "Maintain 5cm standing water during tillering"
        ]
    },
    "maize": {
        "temp": "18°C to 27°C",
        "tips": [
            "Ensure proper nitrogen supply for good yield",
            "Avoid waterlogging, sensitive to excess water",
            "Needs well-drained fertile soil"
        ]
    },
    "chickpea": {
        "temp": "15°C to 30°C",
        "tips": [
            "Drought tolerant, avoid overwatering",
            "Fixes atmospheric nitrogen, low fertilizer needed",
            "Prefers well-drained loamy soil"
        ]
    },
    "kidneybeans": {
        "temp": "18°C to 24°C",
        "tips": [
            "Sensitive to frost, plant after last frost",
            "Avoid high nitrogen, it's a nitrogen fixer",
            "Needs consistent moisture during pod fill"
        ]
    },
    "pigeonpeas": {
        "temp": "18°C to 38°C",
        "tips": [
            "Drought resistant once established",
            "Deep rooted, improves soil structure",
            "Low water requirement compared to other legumes"
        ]
    },
    "mothbeans": {
        "temp": "25°C to 35°C",
        "tips": [
            "Extremely drought tolerant",
            "Grows well in sandy and arid soils",
            "Minimal irrigation required"
        ]
    },
    "mungbean": {
        "temp": "25°C to 35°C",
        "tips": [
            "Short duration crop (60-90 days)",
            "Sensitive to waterlogging",
            "Good green manure crop for soil health"
        ]
    },
    "blackgram": {
        "temp": "25°C to 35°C",
        "tips": [
            "Thrives in humid tropical climate",
            "Nitrogen fixing legume",
            "Avoid waterlogging during flowering"
        ]
    },
    "lentil": {
        "temp": "15°C to 25°C",
        "tips": [
            "Cool season crop, frost tolerant at early stage",
            "Low water requirement",
            "Prefers well-drained neutral pH soil"
        ]
    },
    "pomegranate": {
        "temp": "25°C to 35°C",
        "tips": [
            "Drought tolerant once established",
            "Needs dry weather during ripening",
            "Regular pruning improves yield"
        ]
    },
    "banana": {
        "temp": "20°C to 35°C",
        "tips": [
            "High water and potassium demand",
            "Wind sensitive, needs sheltered location",
            "Needs well-drained fertile soil"
        ]
    },
    "mango": {
        "temp": "24°C to 30°C",
        "tips": [
            "Dry spell before flowering improves fruit set",
            "Avoid frost, highly sensitive",
            "Deep well-drained soil preferred"
        ]
    },
    "grapes": {
        "temp": "15°C to 35°C",
        "tips": [
            "Requires distinct wet and dry seasons",
            "Pruning is critical for good yield",
            "Sensitive to excess humidity (promotes disease)"
        ]
    },
    "watermelon": {
        "temp": "22°C to 30°C",
        "tips": [
            "Needs warm soil, plant after frost",
            "High water demand during fruit development",
            "Sandy loam soil gives best results"
        ]
    },
    "muskmelon": {
        "temp": "24°C to 35°C",
        "tips": [
            "Reduce watering as fruit matures for sweetness",
            "Warm climate essential",
            "Well-drained sandy loam preferred"
        ]
    },
    "apple": {
        "temp": "10°C to 25°C",
        "tips": [
            "Requires chilling hours (cold winter) for dormancy",
            "Well-drained slightly acidic soil",
            "Regular pruning and thinning improves fruit size"
        ]
    },
    "orange": {
        "temp": "15°C to 30°C",
        "tips": [
            "Needs dry spell to improve fruit color",
            "Regular irrigation, avoid drought stress",
            "Sensitive to frost and waterlogging"
        ]
    },
    "papaya": {
        "temp": "22°C to 32°C",
        "tips": [
            "Fast growing, bears fruit in 9-12 months",
            "Cannot tolerate waterlogging",
            "High potassium demand for fruiting"
        ]
    },
    "coconut": {
        "temp": "25°C to 32°C",
        "tips": [
            "High humidity and rainfall preferred",
            "Salt tolerant, grows near coastal areas",
            "Needs 1500mm+ annual rainfall"
        ]
    },
    "cotton": {
        "temp": "20°C to 30°C",
        "tips": [
            "Requires warm climate and moderate rainfall",
            "Needs good sunlight for growth",
            "Avoid excessive moisture during flowering"
        ]
    },
    "jute": {
        "temp": "24°C to 38°C",
        "tips": [
            "Requires high humidity and rainfall",
            "Grows best in alluvial soil",
            "Needs warm and wet climate throughout growth"
        ]
    },
    "coffee": {
        "temp": "15°C to 28°C",
        "tips": [
            "Grows best at high altitudes",
            "Needs shade in early growth stages",
            "Sensitive to frost and waterlogging"
        ]
    }
}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temp = float(request.form['temp'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    soil_type = request.form['soil_type']
    moisture = float(request.form['moisture'])

    # ✅ Match training columns exactly
    crop_input = pd.DataFrame([{
        "n": N,
        "p": P,
        "k": K,
        "temperature": temp,   # was 'temp' — training uses 'temperature'
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    crop = crop_pred_model.predict(crop_input)[0]

    # ✅ Match fertilizer training columns exactly
    fert_input = pd.DataFrame([{
        "temparature": temp,   # note: typo in training data — 'temparature'
        "humidity": humidity,
        "moisture": moisture,
        "nitrogen": N,         # was 'N' — training uses 'nitrogen'
        "potassium": K,        # was 'K' — training uses 'potassium'
        "phosphorous": P,      # was 'P' — training uses 'phosphorous'
        "crop_type": crop,
        "soil_type": soil_type
    }])

    fert = fert_pred_model.predict(fert_input)[0]

    info = crop_advice.get(crop.lower(), None)

    return render_template('index.html', crop=crop, fert=fert, info=info)
    
handler = app.run(debug=True)
