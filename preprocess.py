import pandas as pd

# Load CMAPSS dataset
def load_data():

    columns = [
        'unit',
        'cycle',
        'op_setting1',
        'op_setting2',
        'op_setting3'
    ]

    # Sensor columns
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]

    columns.extend(sensor_cols)

    # Load file
    df = pd.read_csv(
        "data/train_FD001.txt",
        sep=r"\s+",
        header=None
    )

    # Remove extra empty columns
    df = df.iloc[:, :26]

    # Assign column names
    df.columns = columns

    return df


# Create Remaining Useful Life (RUL)
def create_rul(df):

    max_cycle = df.groupby('unit')['cycle'].max().reset_index()

    max_cycle.columns = ['unit', 'max_cycle']

    df = df.merge(max_cycle, on='unit')

    df['RUL'] = df['max_cycle'] - df['cycle']

    df.drop(columns=['max_cycle'], inplace=True)

    return df