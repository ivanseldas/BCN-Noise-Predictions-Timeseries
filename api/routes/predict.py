from fastapi import APIRouter
import mlflow
import pandas as pd

router = APIRouter(prefix="/predict")

MODEL_PATH = "/app/models/production"
model = mlflow.pyfunc.load_model(MODEL_PATH)

# Expected feature order (must match training)
FEATURE_ORDER = [
    "hour", "day_of_week", "month", "year",
    "Latitud", "Longitud", "is_weekend",
    "Lockdown_During lockdown", "Lockdown_Post-lockdown", "Lockdown_Pre-lockdown",
    "public_holiday",
    "Noise_db_lag_1", "Noise_db_lag_24",
    "Noise_db_roll_mean_3", "Noise_db_roll_std_3",
    "Noise_db_roll_mean_24", "Noise_db_roll_std_24",
    "sin_1", "cos_1", "sin_2", "cos_2"
]

@router.post("/")
def predict(data: dict):
    """
    Predict noise level given input features.
    Example:
    {
      "hour": 12,
      "day_of_week": 3,
      "month": 5,
      "year": 2025,
      "public_holiday": 0,
      ...
    }
    """
    # Convert input to DataFrame
    df = pd.DataFrame([data])

    # Ensure all expected features exist
    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0

    # Reorder according to training schema
    df = df[FEATURE_ORDER]

    # Ensure all numeric (avoids dtype=object issues)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Predict
    preds = model.predict(df)

    return {"prediction": preds.tolist(), "features_used": df.to_dict(orient="records")}
