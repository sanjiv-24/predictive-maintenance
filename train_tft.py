import pandas as pd
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE

from lightning.pytorch import Trainer

# -----------------------------
# LOAD DATASET
# -----------------------------

columns = (
    ['engine_id', 'cycle']
    + [f'op_setting_{i}' for i in range(1, 4)]
    + [f'sensor_{i}' for i in range(1, 22)]
)

df = pd.read_csv(
    "data/train_FD001.txt",
    sep=r"\s+",
    header=None
)

# remove extra empty columns if present
df = df.iloc[:, :26]

df.columns = columns

# -----------------------------
# CREATE RUL
# -----------------------------

max_cycle = df.groupby("engine_id")["cycle"].transform("max")

df["RUL"] = max_cycle - df["cycle"]

# -----------------------------
# SENSOR COLUMNS
# -----------------------------

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]

# -----------------------------
# DATASET FOR TFT
# -----------------------------

training = TimeSeriesDataSet(
    df,
    time_idx="cycle",
    target="RUL",
    group_ids=["engine_id"],

    max_encoder_length=30,
    max_prediction_length=1,

    time_varying_known_reals=["cycle"],
    time_varying_unknown_reals=sensor_cols,

    target_normalizer=None,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

train_dataloader = training.to_dataloader(
    train=True,
    batch_size=64,
    num_workers=0
)

# -----------------------------
# TFT MODEL
# -----------------------------

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=16,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=8,

    output_size=1,

    loss=RMSE(),
)

# -----------------------------
# TRAINER
# -----------------------------

trainer = Trainer(
    max_epochs=5,
    accelerator="auto",
    devices=1,
)

# -----------------------------
# TRAIN MODEL
# -----------------------------

trainer.fit(
    tft,
    train_dataloader
)

# -----------------------------
# SAVE MODEL
# -----------------------------

trainer.save_checkpoint("tft_model.ckpt")

print("TFT MODEL TRAINED SUCCESSFULLY")