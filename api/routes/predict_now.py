from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import pandas as pd
import pickle
from typing import List
import mlflow

router = APIRouter(prefix="/predict_now")

MODEL_PATH = "/app/models/production"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError as e:
    raise RuntimeError(f"Model pickle not found at {MODEL_PATH}. Export it first.") from e

# Expected feature order
FEATURE_ORDER = ['hour', 'day_of_week', 'month', 'year','Latitud', 'Longitud']

# Pydantic schemas
class PredictPoint(BaseModel):
    datetime: datetime
    prediction: float

class PredictNowResponse(BaseModel):
    predictions: List[PredictPoint]
    total_points: int
    start: datetime
    end: datetime

class PredictNowRequest(BaseModel):
    lat: float = Field(41.374878, ge=-90, le=90, description="Latitude in degrees")
    lon: float = Field(2.161585, ge=-180, le=180, description="Longitude in degrees")
    hours: int = Field(7 * 24, ge=1, le=7 * 24, description="Number of hourly predictions")

def _build_frame(timestamps: List[datetime], lat: float, lon: float) -> pd.DataFrame:
    records = []
    for ts in timestamps:
        records.append({
            "hour": ts.hour,
            "day_of_week": ts.weekday(),
            "month": ts.month,
            "year": ts.year,
            "Latitud": float(lat),
            "Longitud": float(lon),
            #"is_weekend": 1 if ts.weekday() >= 5 else 0,
            #"Lockdown_During lockdown": 0,
            #"Lockdown_Post-lockdown": 0,
            #"Lockdown_Pre-lockdown": 0,
            #"public_holiday": 0,
            #"Noise_db_lag_1": 0,
            #"Noise_db_lag_24": 0,
            #"Noise_db_roll_mean_3": 0,
            #"Noise_db_roll_std_3": 0,
            #"Noise_db_roll_mean_24": 0,
            #"Noise_db_roll_std_24": 0,
            #"sin_1": 0,
            #"cos_1": 1,
            #"sin_2": 0,
            #"cos_2": 1

        })
    df = pd.DataFrame(records)
    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_ORDER].apply(pd.to_numeric, errors="coerce").fillna(0)
    return df

def _predict_response(lat: float, lon: float, hours: int) -> PredictNowResponse:
    now = datetime.now()
    timestamps = sorted([now - timedelta(hours=i) for i in range(hours)])
    try:
        df = _build_frame(timestamps, lat, lon)
        preds = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    points = [PredictPoint(datetime=ts, prediction=float(p)) for ts, p in zip(timestamps, preds)]
    return PredictNowResponse(
        predictions=points,
        total_points=len(points),
        start=timestamps[0],
        end=timestamps[-1],
    )

@router.get("/", response_model=PredictNowResponse)
def predict_last_week() -> PredictNowResponse:
    # Defaults: Barcelona center and last 7 days (168 hours)
    return _predict_response(lat=41.374878, lon=2.161585, hours=7 * 24)

@router.post("/", response_model=PredictNowResponse)
def predict_custom(req: PredictNowRequest) -> PredictNowResponse:
    return _predict_response(lat=req.lat, lon=req.lon, hours=req.hours)