from flask import Flask, request, jsonify
import joblib
import pandas as pd
import requests
from io import BytesIO

app = Flask(__name__)

# Load model and scaler from Google Drive
def download_file_from_google_drive(url):
    response = requests.get(url)
    return BytesIO(response.content)

model_url = "https://drive.google.com/file/d/1zoegVTXUbPGBZYOP5cuCYbd5ufIJ9M1w/view?usp=sharing"
scaler_url = "https://drive.google.com/file/d/1jAbXY2mCVv53bOr8zfj9Pqupqjz7xfy3/view?usp=sharing"

model = joblib.load(download_file_from_google_drive(model_url))
scaler = joblib.load(download_file_from_google_drive(scaler_url))

@app.route('/')
def home():
    return "Network Anomaly Detection API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
