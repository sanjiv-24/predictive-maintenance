from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
import os

app = FastAPI(
    title="Predictive Maintenance API",
    description="TFT model — NASA CMAPSS FD001",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Supabase client (env vars set in Render dashboard) ──
SUPABASE_URL = os.environ.get("https://tvdsogdicodewiyipmot.supabase.co")
SUPABASE_KEY = os.environ.get("sb_publishable_sZfLoPHHw1JsgM__X4rqPg_PR3S4up7")
db = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.get("/")
def home():
    return {
        "message":   "Predictive Maintenance API is running",
        "model":     "Temporal Fusion Transformer",
        "dataset":   "NASA CMAPSS FD001",
        "endpoints": ["/predict", "/units", "/fleet", "/docs"]
    }

@app.get("/predict")
def predict(unit_id: int = 1, cycle: int = 50):
    # Fetch predictions for this engine, find closest cycle
    res = (
        db.table("predictions")
        .select("*")
        .eq("unit_id", unit_id)
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=404,
                            detail=f"Engine {unit_id} not found")

    # Find closest cycle
    closest = min(res.data, key=lambda x: abs(x['cycle'] - cycle))

    return {
        "unit_id":        closest['unit_id'],
        "cycle":          closest['cycle'],
        "predicted_rul":  closest['predicted_rul'],
        "engine_health":  closest['engine_health'],
        "status":         closest['status'],
    }

@app.get("/units")
def get_units():
    res = db.table("predictions").select("unit_id").execute()
    units = sorted(set(r['unit_id'] for r in res.data))
    return {"total": len(units), "units": units}

@app.get("/fleet")
def fleet(cycle: int = 50, limit: int = 20):
    units_res = db.table("predictions").select("unit_id").execute()
    units = sorted(set(r['unit_id'] for r in units_res.data))[:limit]

    results = []
    for unit_id in units:
        res = (
            db.table("predictions")
            .select("*")
            .eq("unit_id", unit_id)
            .execute()
        )
        if res.data:
            closest = min(res.data, key=lambda x: abs(x['cycle'] - cycle))
            results.append({
                "unit_id":        closest['unit_id'],
                "cycle":          closest['cycle'],
                "predicted_rul":  closest['predicted_rul'],
                "engine_health":  closest['engine_health'],
                "status":         closest['status'],
            })

    statuses = [r['status'] for r in results]
    return {
        "cycle":   cycle,
        "total":   len(results),
        "summary": {
            "CRITICAL": statuses.count("CRITICAL"),
            "WARNING":  statuses.count("WARNING"),
            "NORMAL":   statuses.count("NORMAL"),
        },
        "engines": results
    }