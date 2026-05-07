import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Column names
cols = ['unit', 'cycle'] + [f'op{i}' for i in range(1,4)] + [f's{i}' for i in range(1,22)]

# Load dataset
df = pd.read_csv(
    'data/train_FD001.txt',
    sep='\s+',
    header=None
)

df.columns = cols

# Create Remaining Useful Life (RUL)
max_cycle = df.groupby('unit')['cycle'].max()

df = df.merge(max_cycle.rename('max_cycle'), on='unit')

df['RUL'] = df['max_cycle'] - df['cycle']

# Features
X = df[[f's{i}' for i in range(1,22)]]

# Target
y = df['RUL']

# Train model
model = RandomForestRegressor(n_estimators=50)

model.fit(X, y)

# Save model
joblib.dump(model, 'model.pkl')

print("Model trained and saved as model.pkl")