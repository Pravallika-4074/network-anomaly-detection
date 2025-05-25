from flask import Flask, request, jsonify
import joblib
import pandas as pd
import gdown
import os

app = Flask(__name__)

# Download from Google Drive only once and save locally
def download_if_not_exists(url, output_path):
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

# Use the full gdown shareable link
model_url = "https://drive.google.com/file/d/1zoegVTXUbPGBZYOP5cuCYbd5ufIJ9M1w/view?usp=sharing"
scaler_url = "https://drive.google.com/file/d/1jAbXY2mCVv53bOr8zfj9Pqupqjz7xfy3/view?usp=sharing"

# Define file names to save after download
model_path = "anomalymodel.pkl"
scaler_path = "scaler.pkl"

# Download files if they don't already exist
download_if_not_exists(model_url, model_path)
download_if_not_exists(scaler_url, scaler_path)

# Load the files
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

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
