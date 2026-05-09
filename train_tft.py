import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# ============================================================
# STEP 1 — LOAD DATA
# ============================================================

SENSOR_COLS = [f'sensor_{i}' for i in range(1, 22)]

columns = (
    ['unit', 'cycle', 'op_setting1', 'op_setting2', 'op_setting3']
    + SENSOR_COLS
)

print("Loading dataset...")

df = pd.read_csv(
    "data/train_FD001.txt",
    sep=r"\s+",
    header=None
)

df = df.iloc[:, :26]
df.columns = columns

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# STEP 2 — CREATE RUL LABELS
# ============================================================

max_cycle = df.groupby('unit')['cycle'].transform('max')
df['RUL'] = (max_cycle - df['cycle']).clip(upper=125)

print(f"RUL range: {df['RUL'].min()} to {df['RUL'].max()}")

# ============================================================
# STEP 3 — REMOVE LOW-VARIANCE SENSORS
# ============================================================

# These sensors are constant on FD001 — they hurt training
DROP_SENSORS = ['sensor_1', 'sensor_5', 'sensor_6',
                'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']

GOOD_SENSORS = [s for s in SENSOR_COLS if s not in DROP_SENSORS]

print(f"Using {len(GOOD_SENSORS)} sensors: {GOOD_SENSORS}")

# ============================================================
# STEP 4 — NORMALIZE SENSORS (per-column min-max)
# ============================================================

from sklearn.preprocessing import MinMaxScaler
import joblib

scaler = MinMaxScaler()
df[GOOD_SENSORS] = scaler.fit_transform(df[GOOD_SENSORS])

joblib.dump(scaler, "scaler.pkl")
print("Scaler saved to scaler.pkl")

# ============================================================
# STEP 5 — TRAIN / VAL SPLIT (by engine unit)
# ============================================================

units = df['unit'].unique()
train_units = units[:int(0.8 * len(units))]
val_units   = units[int(0.8 * len(units)):]

train_df = df[df['unit'].isin(train_units)].reset_index(drop=True)
val_df   = df[df['unit'].isin(val_units)].reset_index(drop=True)

print(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows")

# ============================================================
# STEP 6 — BUILD TimeSeriesDataSet
# ============================================================

MAX_ENCODER_LENGTH   = 30
MAX_PREDICTION_LENGTH = 1

training = TimeSeriesDataSet(
    train_df,
    time_idx="cycle",
    target="RUL",
    group_ids=["unit"],
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PREDICTION_LENGTH,
    time_varying_known_reals=["cycle"],
    time_varying_unknown_reals=GOOD_SENSORS,
    target_normalizer=None,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(
    training, val_df, predict=True, stop_randomization=True
)

train_dataloader = training.to_dataloader(
    train=True, batch_size=64, num_workers=0
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=64, num_workers=0
)

print("DataLoaders ready")

# ============================================================
# STEP 7 — BUILD TFT MODEL
# ============================================================

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.003,
    hidden_size=32,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16,
    output_size=1,
    loss=RMSE(),
    log_interval=10,
    reduce_on_plateau_patience=3,
)

print(f"Model parameters: {tft.size() / 1e3:.1f}k")

# ============================================================
# STEP 8 — TRAIN
# ============================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min"
)

checkpoint = ModelCheckpoint(
    dirpath="saved_model",
    filename="best_tft",
    monitor="val_loss",
    save_top_k=1,
    mode="min"
)

trainer = Trainer(
    max_epochs=30,
    accelerator="auto",
    devices=1,
    gradient_clip_val=0.1,
    callbacks=[early_stop, checkpoint],
    enable_progress_bar=True,
)

print("Starting training...")

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# ============================================================
# STEP 9 — SAVE FINAL CHECKPOINT
# ============================================================

trainer.save_checkpoint("tft_model.ckpt")

print("\n========================================")
print("  TFT MODEL TRAINED SUCCESSFULLY")
print(f"  Best val_loss: {trainer.callback_metrics.get('val_loss', 'N/A')}")
print("  Saved: tft_model.ckpt")
print("  Saved: scaler.pkl")
print("========================================\n")

