from fastapi import FastAPI

app = FastAPI()

@app.get("/predict")
def predict():

    return {
        "engine_health": 82,
        "predicted_rul": 47,
        "status": "WARNING",
        "anomaly_score": 0.12
    }