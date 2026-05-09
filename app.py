from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
import torch
import pandas as pd
import os

from preprocess import (
    load_and_prepare,
    get_engine_window,
    get_all_units,
    get_max_cycle,
    GOOD_SENSORS,
)

# ============================================================
# APP INIT
# ============================================================

app = FastAPI(
    title="Predictive Maintenance API",
    description="TFT model inference on NASA CMAPSS FD001 dataset",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# LOAD MODEL + DATA AT STARTUP
# ============================================================

print("\n========================================")
print("  Starting Predictive Maintenance API")
print("========================================")

# Validate files exist
if not os.path.exists("tft_model.ckpt"):
    raise FileNotFoundError("tft_model.ckpt not found. Run train_tft.py first.")

if not os.path.exists("scaler.pkl"):
    raise FileNotFoundError("scaler.pkl not found. Run train_tft.py first.")

print("Loading TFT model...")
MODEL = TemporalFusionTransformer.load_from_checkpoint("tft_model.ckpt")
MODEL.eval()
print("Model loaded successfully!")

print("Loading dataset...")
DF = load_and_prepare()
print(f"Dataset loaded: {DF.shape[0]} rows, {len(get_all_units(DF))} engines")
print("========================================\n")

# ============================================================
# HELPER — RUN INFERENCE
# ============================================================

MAX_ENCODER_LENGTH    = 30
MAX_PREDICTION_LENGTH = 1

def run_inference(unit_id: int, up_to_cycle: int) -> float:
    """
    Builds a TimeSeriesDataSet from the engine window
    and runs TFT prediction. Returns predicted RUL (float).
    """

    window_df = get_engine_window(DF, unit_id, up_to_cycle)

    if len(window_df) < 5:
        raise ValueError(
            f"Engine {unit_id} has only {len(window_df)} cycles "
            f"up to cycle {up_to_cycle}. Need at least 5."
        )

    inference_dataset = TimeSeriesDataSet(
        window_df,
        time_idx="cycle",
        target="RUL",
        group_ids=["unit"],
        max_encoder_length=min(MAX_ENCODER_LENGTH, len(window_df)),
        max_prediction_length=MAX_PREDICTION_LENGTH,
        time_varying_known_reals=["cycle"],
        time_varying_unknown_reals=GOOD_SENSORS,
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    dataloader = inference_dataset.to_dataloader(
        train=False, batch_size=1, num_workers=0
    )

    with torch.no_grad():
        predictions = MODEL.predict(dataloader)

    return max(0.0, float(predictions[0]))


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Predictive Maintenance API is running",
        "model": "Temporal Fusion Transformer",
        "dataset": "NASA CMAPSS FD001",
        "endpoints": ["/predict", "/units", "/fleet", "/docs"]
    }


# ----------------------------------------
# SINGLE ENGINE PREDICTION
# ----------------------------------------

@app.get("/predict")
def predict(unit_id: int = 1, cycle: int = 50):
    """
    Predict Remaining Useful Life for a single engine.

    Args:
        unit_id : engine number (1-100)
        cycle   : simulate up to this cycle number

    Returns:
        cycle, unit_id, predicted_rul, engine_health, status
    """

    # Validate unit_id
    available_units = get_all_units(DF)
    if unit_id not in available_units:
        raise HTTPException(
            status_code=404,
            detail=f"Engine {unit_id} not found. Available: 1-{max(available_units)}"
        )

    # Clamp cycle to valid range
    max_c = get_max_cycle(DF, unit_id)
    cycle = min(cycle, max_c)

    try:
        predicted_rul = run_inference(unit_id, cycle)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Engine health as 0-100 score
    engine_health = round(max(0, min(100, predicted_rul)), 2)

    # Status thresholds
    if predicted_rul < 20:
        status = "CRITICAL"
    elif predicted_rul < 50:
        status = "WARNING"
    else:
        status = "NORMAL"

    return {
        "unit_id":        unit_id,
        "cycle":          cycle,
        "max_cycle":      max_c,
        "predicted_rul":  round(predicted_rul, 2),
        "engine_health":  engine_health,
        "status":         status,
    }


# ----------------------------------------
# ALL AVAILABLE ENGINE UNITS
# ----------------------------------------

@app.get("/units")
def get_units():
    """Returns list of all available engine unit IDs."""
    units = get_all_units(DF)
    return {
        "total": len(units),
        "units": units
    }


# ----------------------------------------
# FLEET OVERVIEW (all engines at a cycle)
# ----------------------------------------

@app.get("/fleet")
def fleet_overview(cycle: int = 50, limit: int = 20):
    """
    Returns RUL predictions for the first `limit` engines
    at a given cycle. Used for the fleet heatmap in the dashboard.

    Args:
        cycle : simulate all engines up to this cycle
        limit : how many engines to return (default 20)
    """

    units = get_all_units(DF)[:limit]
    results = []

    for unit_id in units:
        max_c = get_max_cycle(DF, unit_id)
        safe_cycle = min(cycle, max_c)

        try:
            predicted_rul = run_inference(unit_id, safe_cycle)
            engine_health = round(max(0, min(100, predicted_rul)), 2)

            if predicted_rul < 20:
                status = "CRITICAL"
            elif predicted_rul < 50:
                status = "WARNING"
            else:
                status = "NORMAL"

            results.append({
                "unit_id":       unit_id,
                "cycle":         safe_cycle,
                "predicted_rul": round(predicted_rul, 2),
                "engine_health": engine_health,
                "status":        status,
            })

        except Exception as e:
            results.append({
                "unit_id": unit_id,
                "error":   str(e)
            })

    # Summary counts
    statuses = [r.get("status") for r in results if "status" in r]

    return {
        "cycle":    cycle,
        "total":    len(results),
        "summary": {
            "CRITICAL": statuses.count("CRITICAL"),
            "WARNING":  statuses.count("WARNING"),
            "NORMAL":   statuses.count("NORMAL"),
        },
        "engines": results
    }

