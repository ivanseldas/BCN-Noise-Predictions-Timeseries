# src/data/feature_engineering.py
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.paths import INTERIM_DIR, PROCESSED_DIR

# -------------------------
# 0. Select specific columns
# -------------------------
def select_columns(df: pd.DataFrame, cols_to_keep: list) -> pd.DataFrame:
    """
    Keep only the specified columns from the DataFrame.
    """
    df = df[cols_to_keep].copy()
    return df


# -------------------------
# 1. Time-based features
# -------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based features from the DateTimeIndex.
    """
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
    encoding: str = "label"
) -> pd.DataFrame:
    """
    Add a lockdown feature already encoded.
    encoding = "label" → 0 (Pre), 1 (During), 2 (Post)
    encoding = "onehot" → One-hot encoded columns
    """
    start = pd.to_datetime(lockdown_start)
    end = pd.to_datetime(lockdown_end)

    lockdown_cat = np.where(
        df.index < start, "Pre-lockdown",
        np.where(df.index <= end, "During lockdown", "Post-lockdown")
    )

    if encoding == "label":
        mapping = {"Pre-lockdown": 0, "During lockdown": 1, "Post-lockdown": 2}
        df["Lockdown"] = pd.Series(lockdown_cat).map(mapping).values

    elif encoding == "onehot":
        df = df.copy()
        onehot = pd.get_dummies(lockdown_cat, prefix="Lockdown")
        df = pd.concat([df, onehot.set_index(df.index)], axis=1)

    else:
        raise ValueError("encoding must be 'label' or 'onehot'")

    return df

# -------------------------
# 3. Public holidays feature
# -------------------------
def add_public_holidays_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary feature indicating whether the date is a public holiday in Barcelona.
    Covers years 2015–2024.
    """
    public_holidays = pd.DatetimeIndex([
        # 2015
        "2015-01-01", "2015-01-06", "2015-04-03", "2015-04-06", "2015-05-01",
        "2015-06-24", "2015-08-15", "2015-09-11", "2015-09-24", "2015-10-12",
        "2015-11-01", "2015-12-06", "2015-12-08", "2015-12-25",
        # 2016
        "2016-01-01", "2016-01-06", "2016-03-25", "2016-03-28", "2016-05-01",
        "2016-05-16", "2016-06-24", "2016-08-15", "2016-09-11", "2016-09-24",
        "2016-10-12", "2016-11-01", "2016-12-06", "2016-12-08", "2016-12-25",
        # 2017
        "2017-01-01", "2017-01-06", "2017-04-14", "2017-04-17", "2017-05-01",
        "2017-06-05", "2017-06-24", "2017-08-15", "2017-09-11", "2017-09-24",
        "2017-10-12", "2017-11-01", "2017-12-06", "2017-12-08", "2017-12-25",
        # 2018
        "2018-01-01", "2018-01-06", "2018-03-30", "2018-04-02", "2018-05-01",
        "2018-05-21", "2018-06-24", "2018-08-15", "2018-09-11", "2018-09-24",
        "2018-10-12", "2018-11-01", "2018-12-06", "2018-12-08", "2018-12-25",
        # 2019
        "2019-01-01", "2019-01-06", "2019-04-19", "2019-04-22", "2019-05-01",
        "2019-06-10", "2019-06-24", "2019-08-15", "2019-09-11", "2019-09-24",
        "2019-10-12", "2019-11-01", "2019-12-06", "2019-12-08", "2019-12-25",
        # 2020
        "2020-01-01", "2020-01-06", "2020-04-10", "2020-04-13", "2020-05-01",
        "2020-06-01", "2020-06-24", "2020-08-15", "2020-09-11", "2020-09-24",
        "2020-10-12", "2020-11-01", "2020-12-06", "2020-12-08", "2020-12-25",
        # 2021
        "2021-01-01", "2021-01-06", "2021-04-02", "2021-04-05", "2021-05-01",
        "2021-06-24", "2021-09-11", "2021-09-24", "2021-10-12", "2021-11-01",
        "2021-12-06", "2021-12-08", "2021-12-25",
        # 2022
        "2022-01-01", "2022-01-06", "2022-04-15", "2022-04-18", "2022-06-06",
        "2022-06-24", "2022-08-15", "2022-09-11", "2022-09-24", "2022-10-12",
        "2022-11-01", "2022-12-06", "2022-12-08", "2022-12-25",
        # 2023
        "2023-01-01", "2023-01-06", "2023-04-07", "2023-04-10", "2023-06-24",
        "2023-08-15", "2023-09-11", "2023-09-24", "2023-10-12", "2023-11-01",
        "2023-12-06", "2023-12-08", "2023-12-25",
        # 2024
        "2024-01-01", "2024-01-06", "2024-03-29", "2024-04-01", "2024-05-01",
        "2024-06-24", "2024-08-15", "2024-09-11", "2024-09-24", "2024-10-12",
        "2024-11-01", "2024-12-06", "2024-12-08", "2024-12-25",
    ])

    df["public_holiday"] = df.index.isin(public_holidays).astype(int)
    return df

# -------------------------
# 4. Lag features
# -------------------------
def add_lag_features(df: pd.DataFrame, target_col: str = "Noise_db", lags: list = [1, 24]) -> pd.DataFrame:
    """
    Add lagged versions of the target variable.
    Example: lag 1h, lag 24h.
    """
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


# -------------------------
# 5. Rolling statistics
# -------------------------
def add_rolling_features(df: pd.DataFrame, target_col: str = "Noise_db", windows: list = [3, 24]) -> pd.DataFrame:
    """
    Add rolling mean and std for given window sizes.
    Example: 3h mean/std, 24h mean/std.
    """
    for window in windows:
        df[f"{target_col}_roll_mean_{window}"] = df[target_col].rolling(window=window).mean()
        df[f"{target_col}_roll_std_{window}"] = df[target_col].rolling(window=window).std()
    return df


# -------------------------
# 4. Fourier terms (seasonality)
# -------------------------
def add_fourier_terms(df: pd.DataFrame, period: int = 24, K: int = 2) -> pd.DataFrame:
    """
    Add Fourier terms to model seasonality.
    period = cycle length (24 for daily, 168 for weekly).
    K = number of harmonics.
    """
    t = np.arange(len(df))
    for k in range(1, K + 1):
        df[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        df[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return df


# -------------------------
# 5. Full feature pipeline
# -------------------------
def build_features(df: pd.DataFrame, cols_to_keep: list = None) -> pd.DataFrame:
    """
    Full feature engineering pipeline:
    - Select only required columns
    - Add time-based features
    - Add lockdown feature
    - Add public holidays feature
    - Add lag features
    - Add rolling statistics
    - Add Fourier terms
    """
    if cols_to_keep:
        df = select_columns(df, cols_to_keep)
    df = add_time_features(df)
    df = add_lockdown_feature(df, encoding="onehot")
    df = add_public_holidays_feature(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_fourier_terms(df)
    return df

# Example usage
if __name__ == "__main__":
    # Example standalone run
    df_sensors = pd.read_parquet(INTERIM_DIR / "sensor_496.parquet")
    
    cols_to_keep = ['Id_Instal', 'Noise_db', 'hour', 'day_of_week', 'month', 'year','Latitud', 'Longitud']

    # Build features
    df_features = build_features(df_sensors, cols_to_keep=cols_to_keep)
    print(df_features.head())
    print(df_features.info())
    
    # Save features
    df_features.dropna(inplace=True)  # Drop rows with NaNs from lag/rolling features
    df_features.to_parquet(PROCESSED_DIR / "sensor_496_processed.parquet")
    print(f"Features saved to {PROCESSED_DIR / 'sensor_496_processed.parquet'}")