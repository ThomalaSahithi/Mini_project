from flask import Flask, render_template, request
import numpy as np
import pickle
import logging

# Initialize Flask App
app = Flask(__name__)

# Global variables for models and scaler
rfc = None
gnb = None
ms = None

# Function to load models only when needed
def load_models():
    global rfc, gnb, ms
    if rfc is None or gnb is None or ms is None:
        try:
            with open('model.pkl', 'rb') as f:
                rfc = pickle.load(f)
            with open('gnb_model.pkl', 'rb') as f:
                gnb = pickle.load(f)
            with open('minmaxscaler.pkl', 'rb') as f:
                ms = pickle.load(f)
            app.logger.info("✅ Models loaded successfully.")
        except Exception as e:
            app.logger.error(f"❌ Error loading models: {e}")
            return False
    return True

# Crop Dictionary Mapping
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
    16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Function for Crop Recommendation
def recommendation(N, P, K, temperature, humidity, ph, rainfall, model_type="rfc"):
    if not load_models():
        return None

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    transformed_features = ms.transform(features)

    if model_type == "gnb":
        prediction = gnb.predict(transformed_features)
    else:
        prediction = rfc.predict(transformed_features)

    return prediction[0]

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        crop_code = recommendation(N, P, K, temperature, humidity, ph, rainfall)
        if crop_code is None:
            return "Error loading model. Check logs.", 500

        crop_name = crop_dict.get(crop_code, "Unknown")

        return render_template(
            'result.html',
            N=N, P=P, K=K,
            temperature=temperature,
            humidity=humidity,
            ph=ph, rainfall=rainfall,
            crop=crop_name
        )
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return f"Error: {e}", 400

# Run the Flask App
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
