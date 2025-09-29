from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import pandas as pd
import pickle
from typing import List

router = APIRouter(prefix="/predict_now")

# --- Load model once at import time ---
MODEL_PATH = "models/production/model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError as e:
    raise RuntimeError(f"Model pickle not found at {MODEL_PATH}. Export it first.") from e

# Feature order expected by the trained model
FEATURE_ORDER = ["hour", "day_of_week", "month", "year", "Latitud", "Longitud"]

# Approximate RMSE from validation (adjust with real tracked metric)
RMSE = 2.5
Z_VALUE = 1.96  # 95% confidence (normal approximation)

# ---------- Pydantic Schemas ----------
class PredictPoint(BaseModel):
    datetime: datetime
    prediction: float
    lower: float
    upper: float

class PredictNowResponse(BaseModel):
    predictions: List[PredictPoint]
    total_points: int
    start: datetime
    end: datetime

class PredictNowRequest(BaseModel):
    lat: float = Field(41.374878, ge=-90, le=90, description="Latitude in degrees")
    lon: float = Field(2.161585, ge=-180, le=180, description="Longitude in degrees")
    hours: int = Field(7 * 24, ge=1, le=7 * 24, description="Number of hourly predictions (max 7 days)")

# ---------- Feature frame builder ----------
def _build_frame(timestamps: List[datetime], lat: float, lon: float) -> pd.DataFrame:
    """
    Builds a feature DataFrame for the given timestamps and static latitude/longitude.
    """
    records = []
    for ts in timestamps:
        records.append({
            "hour": ts.hour,
            "day_of_week": ts.weekday(),
            "month": ts.month,
            "year": ts.year,
            "Latitud": float(lat),
            "Longitud": float(lon),
        })
    df = pd.DataFrame(records)
    # Ensure all expected columns exist
    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0
    # Order + numeric coercion
    df = df[FEATURE_ORDER].apply(pd.to_numeric, errors="coerce").fillna(0)
    return df

# ---------- Core prediction logic ----------
def _predict_response(lat: float, lon: float, hours: int) -> PredictNowResponse:
    """
    Generates hourly predictions for the past 'hours' hours up to now.
    (If you want future forecasts, change time direction accordingly.)
    """
    now = datetime.now()
    # Past hours (ascending)
    timestamps = sorted([now - timedelta(hours=i) for i in range(hours)])
    try:
        df = _build_frame(timestamps, lat, lon)
        preds = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    points = []
    for ts, p in zip(timestamps, preds):
        lower = float(p) - Z_VALUE * RMSE
        upper = float(p) + Z_VALUE * RMSE
        points.append(PredictPoint(datetime=ts, prediction=float(p), lower=lower, upper=upper))

    return PredictNowResponse(
        predictions=points,
        total_points=len(points),
        start=timestamps[0],
        end=timestamps[-1],
    )

# ---------- Endpoints ----------
@router.get("/", response_model=PredictNowResponse)
def predict_last_week() -> PredictNowResponse:
    """
    Returns predictions for the previous 7 days (past hours).
    """
    return _predict_response(lat=41.374878, lon=2.161585, hours=7 * 24)

@router.post("/", response_model=PredictNowResponse)
def predict_custom(req: PredictNowRequest) -> PredictNowResponse:
    """
    Returns predictions for a custom location and horizon (past hours).
    """
    return _predict_response(lat=req.lat, lon=req.lon, hours=req.hours)