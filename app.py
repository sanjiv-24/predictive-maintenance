from fastapi import FastAPI
import random

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "Predictive Maintenance API Running Successfully"
    }

@app.get("/predict")
def predict():

    predicted_rul = random.randint(10, 100)

    if predicted_rul < 20:
        status = "CRITICAL"
    elif predicted_rul < 50:
        status = "WARNING"
    else:
        status = "HEALTHY"

    return {
        "engine_health": 100 - predicted_rul,
        "predicted_rul": predicted_rul,
        "status": status,
        "anomaly_score": round(random.uniform(0,1),2)
    }