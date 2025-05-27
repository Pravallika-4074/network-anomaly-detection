from flask import Flask, request, jsonify
import joblib
import pandas as pd
import gdown
import os

app = Flask(__name__)

# Step 1: Define Google Drive URLs
model_url = "https://drive.google.com/uc?id=1X2RJfu8wVvTX5gYN558RMKNt8w1DovT1"
scaler_url = "https://drive.google.com/uc?id=1oY7tER5OTV_SuKe6u5tvOcCRwKn9VWn2"
columns_url = "https://drive.google.com/uc?id=1eBX6_QbIHUHldPhr6tNL3xHOL_AxgRNa"
labelencoder_url = "https://drive.google.com/uc?id=16uQDOP_7os2CC2LGS1i84nd5dCLQ46zf"

# Step 2: Define local paths
model_path = "anomalymodel_v2.pkl"
scaler_path = "scaler_v2.pkl"
columns_path = "columns_used.pkl"
labelencoder_path = "labelencoder.pkl"

# Step 3: Download if not already present
def download_if_not_exists(url, output_path):
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

download_if_not_exists(model_url, model_path)
download_if_not_exists(scaler_url, scaler_path)
download_if_not_exists(columns_url, columns_path)
download_if_not_exists(labelencoder_url, labelencoder_path)

# Step 4: Load the artifacts
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
columns = joblib.load(columns_path)
labelencoder = joblib.load(labelencoder_path)

@app.route('/')
def home():
    return "✅ Network Anomaly Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Ensure correct column order
        df = df.reindex(columns=columns)

        # Encode any object columns (if any exist in test)
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype('category').cat.codes

        # Scale
        df_scaled = scaler.transform(df)

        # Predict
        prediction = model.predict(df_scaled)
        predicted_label = labelencoder.inverse_transform(prediction)

        return jsonify({
            'prediction': int(prediction[0]),
            'predicted_label': predicted_label[0]
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
