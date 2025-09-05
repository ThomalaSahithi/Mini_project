
from  flask import Flask, render_template, request
import numpy as np
import pickle

# Initialize Flask App
app = Flask(__name__)

# Load the trained model and scaler
# Load the models
rfc = pickle.load(open('model.pkl', 'rb'))
gnb = pickle.load(open('gnb_model.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))


# Crop Dictionary Mapping
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
    16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}


# Home Route to Render index.html
@app.route('/')
def home():
    return render_template('index.html')


# Function for Crop Recommendation
def recommendation(N, P, k, temperature, humidity, ph, rainfall, model_type="rfc"):
    features = np.array([[N, P, k, temperature, humidity, ph, rainfall]])

    # Transform using MinMaxScaler
    transformed_features = ms.transform(features)

    # Choose model based on parameter
    if model_type == "gnb":
        prediction = gnb.predict(transformed_features)
    else:
        prediction = rfc.predict(transformed_features)

    return prediction[0]


# Route to Handle Prediction and Show Results
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Get prediction
    crop_code = recommendation(N, P, K, temperature, humidity, ph, rainfall)
    crop_name = crop_dict.get(crop_code, "Unknown")

    # Render result page with entered details and predicted output
    return render_template(
        'result.html',
        N=N, P=P, K=K,
        temperature=temperature,
        humidity=humidity,
        ph=ph, rainfall=rainfall,
        crop=crop_name
    )


# Run the Flask App
if __name__ == '__main__':
    app.run(debug=True)
