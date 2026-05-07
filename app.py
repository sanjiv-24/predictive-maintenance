from fastapi import FastAPI
import joblib
import numpy as np

# Create FastAPI app
app = FastAPI()

# Load trained model
model = joblib.load("model.pkl")

# Home route
@app.get("/")
def home():
    return {"message": "Predictive Maintenance API Running"}

# Prediction route
@app.post("/predict")
def predict(data: dict):

    # Extract sensor values
    sensor_values = data["sensors"]

    # Convert into numpy array
    features = np.array(sensor_values).reshape(1, -1)

    # Predict RUL
    prediction = model.predict(features)

    return {
        "Predicted_RUL": float(prediction[0])
    }