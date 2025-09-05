# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
print("Loading dataset...")
crop = pd.read_csv("Crop_recommendation.csv")
crop = crop.loc[:, ~crop.columns.str.contains('^Unnamed')]
print(crop.columns)


# Check for missing values and drop unnecessary columns
print("Checking for missing values:")
print(crop.isnull().sum())
crop = crop.drop(columns=['Unnamed: 8', 'Unnamed: 9'], errors='ignore')

# Check for duplicates and drop them if any
print("Checking for duplicates:", crop.duplicated().sum())
crop.drop_duplicates(inplace=True)

# Define the target dictionary for encoding crop labels
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
    'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10,
    'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15,
    'blackgram': 16, 'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19,
    'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}

# Map labels to numeric values
crop['crop_num'] = crop['label'].map(crop_dict)

# Define features and target
X = crop[['Nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']]
y = crop['crop_num']

# Split the dataset into training and testing sets
print("Splitting dataset into training and testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using MinMaxScaler
print("Applying MinMaxScaler...")
ms = MinMaxScaler()
X_train_scaled = ms.fit_transform(X_train)
X_test_scaled = ms.transform(X_test)

# Train Random Forest Classifier
print("Training Random Forest Classifier...")
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train_scaled, y_train)

# Train Gaussian Naive Bayes Classifier
print("Training Gaussian Naive Bayes Classifier...")
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)

# Evaluate the models
from sklearn.metrics import accuracy_score

y_pred_rfc = rfc.predict(X_test_scaled)
y_pred_gnb = gnb.predict(X_test_scaled)

print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rfc):.4f}")
print(f"Gaussian Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_gnb):.4f}")

# Save the models and scaler using pickle
print("Saving models and scaler...")
pickle.dump(rfc, open('model.pkl', 'wb'))
pickle.dump(gnb, open('gnb_model.pkl', 'wb'))
pickle.dump(ms, open('minmaxscaler.pkl', 'wb'))

print("Model training and saving complete!")
