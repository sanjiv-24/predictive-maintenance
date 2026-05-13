from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from dotenv import load_dotenv
import os
import uvicorn

# ─────────────────────────────────────────────
# Load Environment Variables
# ─────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Predictive Maintenance API",
    description="TFT model — NASA CMAPSS FD001",
    version="2.0.0"
)

# ─────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Supabase Configuration
# ─────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Debug Prints
print("SUPABASE_URL =", SUPABASE_URL)
print("SUPABASE_KEY =", "Loaded ✅" if SUPABASE_KEY else "Missing ❌")

# Validate
if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL environment variable is missing")

if not SUPABASE_KEY:
    raise ValueError("SUPABASE_KEY environment variable is missing")

# Create Supabase Client
db = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─────────────────────────────────────────────
# Home Route
# ─────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "Predictive Maintenance API is running",
        "model": "Temporal Fusion Transformer",
        "dataset": "NASA CMAPSS FD001",
        "endpoints": [
            "/predict",
            "/units",
            "/fleet",
            "/docs"
        ]
    }

# ─────────────────────────────────────────────
# Predict Endpoint
# ─────────────────────────────────────────────
@app.get("/predict")
def predict(unit_id: int = 1, cycle: int = 50):

    res = (
        db.table("predictions")
        .select("*")
        .eq("unit_id", unit_id)
        .execute()
    )

    if not res.data:
        raise HTTPException(
            status_code=404,
            detail=f"Engine {unit_id} not found"
        )

    closest = min(
        res.data,
        key=lambda x: abs(x["cycle"] - cycle)
    )

    return {
        "unit_id": closest["unit_id"],
        "cycle": closest["cycle"],
        "predicted_rul": closest["predicted_rul"],
        "engine_health": closest["engine_health"],
        "status": closest["status"]
    }

# ─────────────────────────────────────────────
# Units Endpoint
# ─────────────────────────────────────────────
@app.get("/units")
def get_units():

    res = (
        db.table("predictions")
        .select("unit_id")
        .execute()
    )

    units = sorted(
        set(r["unit_id"] for r in res.data)
    )

    return {
        "total": len(units),
        "units": units
    }

# ─────────────────────────────────────────────
# Fleet Endpoint
# ─────────────────────────────────────────────
@app.get("/fleet")
def fleet(cycle: int = 50, limit: int = 20):

    units_res = (
        db.table("predictions")
        .select("unit_id")
        .execute()
    )

    units = sorted(
        set(r["unit_id"] for r in units_res.data)
    )[:limit]

    results = []

    for unit_id in units:

        res = (
            db.table("predictions")
            .select("*")
            .eq("unit_id", unit_id)
            .execute()
        )

        if res.data:

            closest = min(
                res.data,
                key=lambda x: abs(x["cycle"] - cycle)
            )

            results.append({
                "unit_id": closest["unit_id"],
                "cycle": closest["cycle"],
                "predicted_rul": closest["predicted_rul"],
                "engine_health": closest["engine_health"],
                "status": closest["status"]
            })

    statuses = [r["status"] for r in results]

    return {
        "cycle": cycle,
        "total": len(results),

        "summary": {
            "CRITICAL": statuses.count("CRITICAL"),
            "WARNING": statuses.count("WARNING"),
            "NORMAL": statuses.count("NORMAL")
        },

        "engines": results
    }

# ─────────────────────────────────────────────
# Run Server
# ─────────────────────────────────────────────
if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        reload=True
    )