from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
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

df.columns = columns

# Create RUL
max_cycle = df.groupby('engine_id')['cycle'].max()

df['RUL'] = df.apply(
    lambda row: max_cycle[row['engine_id']] - row['cycle'],
    axis=1
)

# Features
X = df.drop(columns=['RUL'])
y = df['RUL']

# Train split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# Train
model = RandomForestRegressor()

model.fit(X_train, y_train)

print("MODEL TRAINED")