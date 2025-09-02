# 🌱 Crop Prediction API

This project provides a Flask-based REST API that predicts the most suitable crop using **LSTM** and **CNN-LSTM** deep learning models.  
It takes soil and environmental features as input, processes them, and returns the predicted crops.

---

## 🚀 Getting Started

### 📦 API Endpoint
POST /crops/predict/
### Request Body (JSON)
```bash
{
  "features": [N, P, K, temperature, humidity, ph, rainfall]
}
```
### 📤 Example Usage(Using Python)
```bash
import requests

url = "http://127.0.0.1:5000/crops/predict/"
data = {"features": [90, 42, 43, 20.87, 82.0, 6.5, 202.9]}

response = requests.post(url, json=data)
print(response.json())
```
### 📥 Example Response
```bash
{
  "lstm_prediction": "rice",
  "cnn_lstm_prediction": "wheat"
}

```
