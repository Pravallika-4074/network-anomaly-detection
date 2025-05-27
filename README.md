# network-anomaly-detection

Objective - With increasing cybersecurity threats, it's crucial to detect abnormal network traffic that may signal intrusions or attacks. The goal is to identify whether a network connection is normal or anomalous based on connection-level features 
To build and deploy a classification model that detects malicious network behavior in real time.

Target Metric
Accuracy
Precision, Recall, F1-Score for anomaly class
Deployment responsiveness

Key Insights from EDA
Most attacks occur via tcp and http.
Specific services like telnet and private have high anomaly rates.
Duration and wrong_fragment features are strong indicators.

Final Model Performance
Metric	Value
Accuracy	0.97
Precision	0.94
Recall	0.95
F1 Score	0.94

Model used: RandomForestClassifier

Files in this Repository
File	Description
network_anomaly_detection.ipynb	Full notebook: EDA, preprocessing, modeling
app.py	Flask app for deployment
requirements.txt	Required Python libraries
README.md	Project summary & API info

Deployment
### Live API Endpoint:
[(https://network-anomaly-detection-0fjq.onrender.com)]

 Method: POST
 Headers: Content-Type: application/json
 Input Sample:

json
Copy
Edit
{
  "duration": 0,
  "protocol_type": "tcp",
  "service": "http",
  "src_bytes": 181,
  "dst_bytes": 5450,
  ...
}
Output Sample:

json
Copy
Edit
{
  "prediction": 11,
  "label": "normal"
}

External Resources
## Model Artifacts (Download Links)

- [anomalymodel_v2.pkl](https://drive.google.com/uc?id=1X2RJfu8wVvTX5gYN558RMKNt8w1DovT1)
- [scaler_v2.pkl](https://drive.google.com/uc?id=1oY7tER5OTV_SuKe6u5tvOcCRwKn9VWn2)
- [columns_used.pkl](https://drive.google.com/uc?id=1eBX6_QbIHUHldPhr6tNL3xHOL_AxgRNa)
- [labelencoder.pkl](https://drive.google.com/uc?id=16uQDOP_7os2CC2LGS1i84nd5dCLQ46zf)






