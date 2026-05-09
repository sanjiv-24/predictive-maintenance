from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI()

# -----------------------------
# CORS CONFIGURATION
# -----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# GLOBAL VARIABLES
# -----------------------------

cycle = 0
current_rul = 100

# -----------------------------
# HOME ROUTE
# -----------------------------

@app.get("/")
def home():

    return {
        "message": "Predictive Maintenance API Running"
    }

# -----------------------------
# PREDICTION ROUTE
# -----------------------------

@app.get("/predict")
def predict():

    global cycle
    global current_rul

    # Increase cycle
    cycle += 1

    # Reduce RUL gradually
    current_rul -= random.randint(1, 3)

    # Prevent negative values
    if current_rul < 0:
        current_rul = 0

    predicted_rul = current_rul

    # Engine health
    engine_health = max(
        0,
        min(100, predicted_rul)
    )

    # Status logic
    if predicted_rul < 20:
        status = "CRITICAL"

    elif predicted_rul < 50:
        status = "WARNING"

    else:
        status = "NORMAL"

    return {

        "cycle": cycle,

        "engine_health": engine_health,

        "predicted_rul": predicted_rul,

        "status": status,

        "anomaly_score": round(
            random.uniform(0, 1),
            2
        )
    }