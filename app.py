from flask import Flask, render_template, request
import numpy as np
import pickle
import logging

app = Flask(__name__)

# Global variables for models and scaler
rfc = None
ms = None

# Load models when needed
def load_models():
    global rfc, ms
    if rfc is None or ms is None:
        try:
            with open('model.pkl', 'rb') as f:
                rfc = pickle.load(f)
            with open('minmaxscaler.pkl', 'rb') as f:
                ms = pickle.load(f)
            app.logger.info("✅ Models loaded successfully.")
        except Exception as e:
            app.logger.error(f"❌ Error loading models: {e}")
            return False
    return True

# Crop mapping
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
    16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# ✅ Input validation function
def validate_input(N, P, K, temperature, humidity, ph, rainfall):
    if all(v == 0 for v in [N, P, K, temperature, humidity, ph, rainfall]):
        return "❌ All input values are 0 — no crop can grow!"

    if not (0 <= N <= 140):
        return "⚠️ Nitrogen (N) must be between 0 and 140."
    if not (0 <= P <= 145):
        return "⚠️ Phosphorus (P) must be between 0 and 145."
    if not (0 <= K <= 205):
        return "⚠️ Potassium (K) must be between 0 and 205."
    if not (0 <= temperature <= 50):
        return "⚠️ Temperature must be between 0°C and 50°C."
    if not (10 <= humidity <= 100):
        return "⚠️ Humidity must be between 10% and 100%."
    if not (3.5 <= ph <= 9):
        return "⚠️ pH must be between 3.5 and 9."
    if not (20 <= rainfall <= 500):
        return "⚠️ Rainfall must be between 20 and 500 mm."

    return None  # valid input

@app.route('/')
def home():
    return render_template('index.html')

# ✅ Recommend top 5 crops
def recommend_crops(N, P, K, temperature, humidity, ph, rainfall):
    if not load_models():
        return None

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    transformed = ms.transform(features)

    probs = rfc.predict_proba(transformed)[0]
    top_indices = np.argsort(probs)[::-1][:5]
    top_crops = [crop_dict.get(i + 1, "Unknown") for i in top_indices]
    return top_crops

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

        # Validate input
        error = validate_input(N, P, K, temperature, humidity, ph, rainfall)
        if error:
            return render_template('result.html', error=error)

        top_crops = recommend_crops(N, P, K, temperature, humidity, ph, rainfall)
        if top_crops is None:
            return "❌ Error loading model.", 500

        return render_template(
            'result.html',
            N=N, P=P, K=K,
            temperature=temperature, humidity=humidity,
            ph=ph, rainfall=rainfall,
            crops=top_crops
        )

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return f"Error: {e}", 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
