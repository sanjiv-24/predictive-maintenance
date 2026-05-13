# push_predictions.py
# Run locally to generate predictions and push to Supabase

import pandas as pd
import torch

from supabase import create_client
from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet
)

from preprocess import (
    load_and_prepare,
    get_engine_window,
    GOOD_SENSORS
)

# ─────────────────────────────────────────────────────
# Supabase connection
# ─────────────────────────────────────────────────────

SUPABASE_URL = "https://tvdsogdicodewiyipmot.supabase.co/rest/v1/"

SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2ZHNvZ2RpY29kZXdpeWlwbW90Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzgzMzQ1NTUsImV4cCI6MjA5MzkxMDU1NX0.ABQQmJP-Rq3ycKHBqkU7nAxpGDdshlFYSJth0LIQVT8"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─────────────────────────────────────────────────────
# Load TFT model
# ─────────────────────────────────────────────────────

print("Loading model...")

MODEL = TemporalFusionTransformer.load_from_checkpoint(
    "tft_model.ckpt"
)

MODEL.eval()

# ─────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────

print("Loading data...")

df = load_and_prepare()

units = sorted(df["unit"].unique().tolist())[:20]

# ─────────────────────────────────────────────────────
# Run inference
# ─────────────────────────────────────────────────────

results = []

for unit_id in units:

    max_c = int(
        df[df["unit"] == unit_id]["cycle"].max()
    )

    for cycle in range(30, max_c, 5):

        window_df = get_engine_window(
            df,
            unit_id,
            cycle
        )

        # Skip very tiny windows
        if len(window_df) < 6:
            continue

        try:

            encoder_len = min(
                20,
                len(window_df) - 1
            )

            if encoder_len < 5:
                continue

            ds = TimeSeriesDataSet(

                window_df,

                time_idx="cycle",
                target="RUL",
                group_ids=["unit"],

                min_encoder_length=5,
                max_encoder_length=encoder_len,

                min_prediction_length=1,
                max_prediction_length=1,

                time_varying_known_reals=[
                    "cycle"
                ],

                time_varying_unknown_reals=GOOD_SENSORS,

                target_normalizer=None,

                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
            )

            dl = ds.to_dataloader(
                train=False,
                batch_size=1,
                num_workers=0
            )

            with torch.no_grad():
                pred = MODEL.predict(dl)

            rul = max(
                0.0,
                float(pred[0])
            )

            status = (
                "CRITICAL"
                if rul < 20 else
                "WARNING"
                if rul < 50 else
                "NORMAL"
            )

            results.append({
                "unit_id": unit_id,
                "cycle": cycle,
                "predicted_rul": round(rul, 2),
                "engine_health": round(min(100, rul), 2),
                "status": status
            })

            print(
                f"Engine {unit_id:3d} | "
                f"Cycle {cycle:3d} | "
                f"RUL {rul:.1f} | "
                f"{status}"
            )

        except Exception as e:

            print(
                f"Skipped Engine {unit_id} "
                f"Cycle {cycle} → {e}"
            )

# ─────────────────────────────────────────────────────
# Push predictions to Supabase
# ─────────────────────────────────────────────────────

print(f"\nPushing {len(results)} predictions...")

# Delete old predictions
supabase.table("predictions") \
    .delete() \
    .neq("id", 0) \
    .execute()

# Insert in batches
batch_size = 100

for i in range(0, len(results), batch_size):

    batch = results[i:i + batch_size]

    supabase.table("predictions") \
        .insert(batch) \
        .execute()

    print(
        f"Pushed batch "
        f"{i // batch_size + 1}"
    )

print("\nDone!")
print("All predictions are now in Supabase.")