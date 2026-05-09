from fastapi import FastAPI
import random

app = FastAPI()

@app.get("/")
def home():

    return {
        "message": "Predictive Maintenance API Running"
    }

@app.get("/predict")
def predict():

    predicted_rul = random.randint(10, 100)

    # Better health logic
    engine_health = min(100, predicted_rul)

    if predicted_rul < 20:
        status = "CRITICAL"
    elif predicted_rul < 50:
        status = "WARNING"
    else:
        status = "HEALTHY"

    return {
        "engine_health": engine_health,
        "predicted_rul": predicted_rul,
        "status": status,
        "anomaly_score": round(random.uniform(0,1),2)
    }