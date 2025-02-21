from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd

#Load the saved models
lstm_model = tf.keras.models.load_model("lstm_model.h5")
cnn_lstm_model = tf.keras.models.load_model("cnn_lstm_model.h5")

#Load the label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

#Load the dataset to fit scaler
data = pd.read_csv("Crop_recommendation.csv")
X = data.drop("label", axis=1)
scaler = StandardScaler().fit(X)

app = Flask(__name__)

@app.route("/crops/predict/", methods=["POST"])
def predict():
    try:
        #Get input JSON data
        input_data = request.get_json()
        features = np.array(input_data["features"]).reshape(1, -1)
        
        #Standardize the input
        features_scaled = scaler.transform(features)
        
        #Reshape for LSTM and CNN-LSTM models
        features_lstm = features_scaled.reshape(1, 1, -1)
        features_cnn_lstm = features_scaled.reshape(1, -1, 1)
        
        #Make predictions
        lstm_pred = np.argmax(lstm_model.predict(features_lstm))
        cnn_lstm_pred = np.argmax(cnn_lstm_model.predict(features_cnn_lstm))
        
        #Decode predictions
        lstm_crop = label_encoder.inverse_transform([lstm_pred])[0]
        cnn_lstm_crop = label_encoder.inverse_transform([cnn_lstm_pred])[0]
        
        return jsonify({
            "lstm_prediction": lstm_crop,
            "cnn_lstm_prediction": cnn_lstm_crop
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
