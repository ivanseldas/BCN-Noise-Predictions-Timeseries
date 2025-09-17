from fastapi import APIRouter
import mlflow
import pandas as pd
from datetime import datetime, timedelta

router = APIRouter(prefix="/predict_now")

MODEL_PATH = "/app/models/production"
model = mlflow.pyfunc.load_model(MODEL_PATH)

# Expected feature order
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

@router.get("/")
def predict_last_week():
    """
    Predict noise levels for the last 7 days (hourly, 168 points).
    """
    now = datetime.now()
    timestamps = [now - timedelta(hours=i) for i in range(7 * 24)]
    timestamps = sorted(timestamps)  # chronological order

    # Build list of feature dicts
    records = []
    for ts in timestamps:
        input_data = {
            "hour": ts.hour,
            "day_of_week": ts.weekday(),
            "month": ts.month,
            "year": ts.year,
            "Latitud": 41.374878,
            "Longitud": 2.161585,
            "is_weekend": 1 if ts.weekday() >= 5 else 0,
            "Lockdown_During lockdown": 0,
            "Lockdown_Post-lockdown": 0,
            "Lockdown_Pre-lockdown": 0,
            "public_holiday": 0,
            "Noise_db_lag_1": 0,
            "Noise_db_lag_24": 0,
            "Noise_db_roll_mean_3": 0,
            "Noise_db_roll_std_3": 0,
            "Noise_db_roll_mean_24": 0,
            "Noise_db_roll_std_24": 0,
            "sin_1": 0,
            "cos_1": 1,
            "sin_2": 0,
            "cos_2": 1
        }
        records.append(input_data)

    # Create DataFrame in correct order
    df = pd.DataFrame(records)
    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_ORDER].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Batch prediction
    preds = model.predict(df)

    # Combine with timestamps
    output = [
        {"datetime": ts.strftime("%Y-%m-%d %H:%M:%S"), "prediction": float(pred)}
        for ts, pred in zip(timestamps, preds)
    ]

    return {
        "predictions": output,
        "total_points": len(output),
        "start": timestamps[0].strftime("%Y-%m-%d %H:%M:%S"),
        "end": timestamps[-1].strftime("%Y-%m-%d %H:%M:%S")
    }
