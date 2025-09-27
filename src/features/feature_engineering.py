# src/data/feature_engineering.py
import pandas as pd
import numpy as np
import hashlib, json
from pathlib import Path
from src.utils.paths import INTERIM_DIR, PROCESSED_DIR


DEFAULT_COLUMNS =['Id_Instal', 'Noise_db', 'hour', 'day_of_week', 'month', 'year','Latitud', 'Longitud']

# -------------------------
# 1. Time-based features
# -------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features from the DateTimeIndex."""
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df.index.month
    return df


# -------------------------
# 2. Lockdown period feature
# -------------------------
def add_lockdown_feature(
    df: pd.DataFrame,
    lockdown_start: str = "2020-03-14",
    lockdown_end: str = "2020-06-07",
    encoding: str = "label",
) -> pd.DataFrame:
    """Add lockdown feature: label (0/1/2) or one-hot encoding."""
    start = pd.to_datetime(lockdown_start)
    end = pd.to_datetime(lockdown_end)

    lockdown_cat = np.where(
        df.index < start, "Pre-lockdown",
        np.where(df.index <= end, "During lockdown", "Post-lockdown"),
    )

    if encoding == "label":
        mapping = {"Pre-lockdown": 0, "During lockdown": 1, "Post-lockdown": 2}
        df["Lockdown"] = pd.Series(lockdown_cat).map(mapping).values
    elif encoding == "onehot":
        onehot = pd.get_dummies(lockdown_cat, prefix="Lockdown")
        df = pd.concat([df, onehot.set_index(df.index)], axis=1)
    else:
        raise ValueError("encoding must be 'label' or 'onehot'")

    return df


# -------------------------
# 3. Public holidays feature
# -------------------------
def add_public_holidays_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary feature indicating whether the date is a public holiday in Barcelona (2015–2024)."""
    public_holidays = pd.DatetimeIndex([
        "2020-01-01", "2020-01-06", "2020-04-10", "2020-04-13", "2020-05-01",
        "2020-06-01", "2020-06-24", "2020-08-15", "2020-09-11", "2020-09-24",
        "2020-10-12", "2020-11-01", "2020-12-06", "2020-12-08", "2020-12-25",
    ])
    df["public_holiday"] = df.index.isin(public_holidays).astype(int)
    return df


# -------------------------
# 4. Lag features
# -------------------------
def add_lag_features(df: pd.DataFrame, target_col: str = "Noise_db", lags: list = [1, 24]) -> pd.DataFrame:
    """Add lagged versions of the target variable (e.g., lag 1h, lag 24h)."""
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


# -------------------------
# 5. Rolling statistics
# -------------------------
def add_rolling_features(df: pd.DataFrame, target_col: str = "Noise_db", windows: list = [3, 24]) -> pd.DataFrame:
    """Add rolling mean and std for given window sizes."""
    for window in windows:
        df[f"{target_col}_roll_mean_{window}"] = df[target_col].rolling(window=window).mean()
        df[f"{target_col}_roll_std_{window}"] = df[target_col].rolling(window=window).std()
    return df


# -------------------------
# 6. Fourier terms (seasonality)
# -------------------------
def add_fourier_terms(df: pd.DataFrame, period: int = 24, K: int = 2) -> pd.DataFrame:
    """Add Fourier terms to model seasonality (daily, weekly, etc.)."""
    t = np.arange(len(df))
    for k in range(1, K + 1):
        df[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        df[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return df


# -------------------------
# 7. Full feature pipeline
# -------------------------
def build_features(
    df: pd.DataFrame,
    cols_to_keep: list = None,
    lags: list = [1, 24],
    windows: list = [3, 24],
    fourier_period: int = 24,
    fourier_K: int = 2,
    lockdown_encoding: str = "onehot",
) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = add_time_features(df)
    df = add_lockdown_feature(df, encoding=lockdown_encoding)
    df = add_public_holidays_feature(df)
    df = add_lag_features(df, lags=lags)
    df = add_rolling_features(df, windows=windows)
    df = add_fourier_terms(df, period=fourier_period, K=fourier_K)

    # ✅ Apply filtering at the end, only keep existing cols
    if cols_to_keep:
        df = df[[c for c in cols_to_keep if c in df.columns]]

    return df


# -------------------------
# 8. Build features + hash
# -------------------------
def build_features_with_hash(
    df: pd.DataFrame,
    cols_to_keep: list = DEFAULT_COLUMNS,
    lags: list = [1, 24],
    windows: list = [3, 24],
    fourier_period: int = 24,
    fourier_K: int = 2,
    lockdown_encoding: str = "onehot",
):
    """Build features and return (df_features, feature_hash, feature_config)."""
    df_features = build_features(
        df,
        cols_to_keep=cols_to_keep,
        lags=lags,
        windows=windows,
        fourier_period=fourier_period,
        fourier_K=fourier_K,
        lockdown_encoding=lockdown_encoding,
    )

    feature_config = {
        "columns": cols_to_keep,
        "lags": lags,
        "windows": windows,
        "fourier": {"period": fourier_period, "K": fourier_K},
        "lockdown": lockdown_encoding,
        "holidays": True,
    }

    canonical = json.dumps(feature_config, sort_keys=True)
    feature_hash = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]

    return df_features, feature_hash, feature_config


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    df_sensors = pd.read_parquet(INTERIM_DIR / "sensor_496.parquet")

    df_features, fhash, fconfig = build_features_with_hash(
        df_sensors,
        cols_to_keep=DEFAULT_COLUMNS,
        lags=[1, 24],
        windows=[3, 24],
        fourier_period=24,
        fourier_K=2,
    )

    print(df_features.head(), df_features.shape)
    print("Feature hash:", fhash)
    print("Feature config:", fconfig)

    save_path = PROCESSED_DIR / "sensor_496_processed.parquet"
    df_features.dropna(inplace=True)
    df_features.to_parquet(save_path)
    print(f"Features saved to {save_path}")
