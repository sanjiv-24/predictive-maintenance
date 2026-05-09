from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

from preprocess import load_data, create_rul

# Load dataset
df = load_data()

# Create RUL labels
df = create_rul(df)

# Features
X = df.drop(columns=['RUL'])

# Target
y = df['RUL']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=50,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
mae = mean_absolute_error(y_test, predictions)

print(f"MAE: {mae}")

# Create folder if not exists
os.makedirs("saved_model", exist_ok=True)

# Save model
joblib.dump(model, "saved_model/rf_model.pkl")

print("MODEL SAVED")