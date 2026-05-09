import pandas as pd
import joblib
import os

# ============================================================
# CONSTANTS — must match train_tft.py exactly
# ============================================================

SENSOR_COLS = [f'sensor_{i}' for i in range(1, 22)]

DROP_SENSORS = ['sensor_1', 'sensor_5', 'sensor_6',
                'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']

GOOD_SENSORS = [s for s in SENSOR_COLS if s not in DROP_SENSORS]

COLUMNS = (
    ['unit', 'cycle', 'op_setting1', 'op_setting2', 'op_setting3']
    + SENSOR_COLS
)

MAX_ENCODER_LENGTH = 30

# ============================================================
# LOAD SCALER
# ============================================================

def load_scaler():
    """Load the MinMaxScaler saved during training."""
    if not os.path.exists("scaler.pkl"):
        raise FileNotFoundError(
            "scaler.pkl not found. Run train_tft.py first."
        )
    return joblib.load("scaler.pkl")


# ============================================================
# LOAD FULL DATASET
# ============================================================

def load_and_prepare(filepath="data/train_FD001.txt"):
    """
    Load NASA CMAPSS FD001 dataset.
    Returns a cleaned DataFrame with RUL column and normalized sensors.
    """

    df = pd.read_csv(filepath, sep=r"\s+", header=None)
    df = df.iloc[:, :26]
    df.columns = COLUMNS

    # Compute RUL
    max_cycle = df.groupby('unit')['cycle'].transform('max')
    df['RUL'] = (max_cycle - df['cycle']).clip(upper=125)

    # Normalize sensors using saved scaler
    scaler = load_scaler()
    df[GOOD_SENSORS] = scaler.transform(df[GOOD_SENSORS])

    return df


# ============================================================
# GET ENGINE WINDOW FOR INFERENCE
# ============================================================

def get_engine_window(df, unit_id: int, up_to_cycle: int,
                      window: int = MAX_ENCODER_LENGTH):
    """
    Returns the last `window` cycles of sensor data
    for a given engine, up to `up_to_cycle`.

    Args:
        df         : full prepared DataFrame from load_and_prepare()
        unit_id    : engine unit number (1-100 in FD001)
        up_to_cycle: simulate reading up to this cycle
        window     : how many past cycles to include

    Returns:
        DataFrame with the sensor window (used for TFT inference)
    """

    engine_df = df[df['unit'] == unit_id].copy()

    window_df = (
        engine_df[engine_df['cycle'] <= up_to_cycle]
        .tail(window)
        .reset_index(drop=True)
    )

    return window_df


# ============================================================
# GET AVAILABLE UNITS
# ============================================================

def get_all_units(df):
    """Returns sorted list of all engine unit IDs."""
    return sorted(df['unit'].unique().tolist())


def get_max_cycle(df, unit_id: int):
    """Returns the last cycle number for a given engine."""
    return int(df[df['unit'] == unit_id]['cycle'].max())


# ============================================================
# QUICK TEST (run this file directly to verify)
# ============================================================

if __name__ == "__main__":
    print("Testing preprocess.py...")

    df = load_and_prepare()
    print(f"Full dataset shape: {df.shape}")
    print(f"Available units: {len(get_all_units(df))}")
    print(f"RUL range: {df['RUL'].min()} - {df['RUL'].max()}")
    print(f"Good sensors used: {GOOD_SENSORS}")

    # Test window for engine 1 at cycle 50
    window = get_engine_window(df, unit_id=1, up_to_cycle=50)
    print(f"\nEngine 1 window at cycle 50: {window.shape}")
    print(window[['unit', 'cycle', 'RUL'] + GOOD_SENSORS[:3]].tail())

    print("\npreprocess.py OK")

