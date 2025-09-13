import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pathlib import Path
from src.utils.paths import SENSORS_RAW_FOLDER, MEASURES_RAW_FOLDER, INTERIM_DIR
from src.data.load_data import load_raw_data, merge_datasets



# -------------------------
# 1. Build timestamp, rename target, sort
# -------------------------
def initial_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Combine the date and time columns into a single DateTimeIndex
    df["Datetime"] = pd.to_datetime(df[["Any", "Mes", "Dia"]].astype(str).agg('-'.join, axis=1) + " " + df["Hora"])
    df.set_index("Datetime", inplace=True)

    # Drop the original columns used to create the DateTimeIndex
    df.drop(columns=["Any", "Mes", "Dia", "Hora"], inplace=True)

    # Change name of Noise db data
    df.rename(columns={'Nivell_LAeq_1h':'Noise_db'}, inplace=True)

    # Sort dataframe by index
    df.sort_index(inplace=True)

    return df

# -------------------------
# 2. Reindex time series
# -------------------------
def reindex_with_full_range(df: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """
    Ensure the DataFrame covers the full date range without gaps.
    Missing timestamps will be added with NaNs.
    """
    start_date = df.index.min()
    end_date = df.index.max()
    full_range = pd.date_range(start=start_date, end=end_date, freq=freq)

    df_reindexed = df.reindex(full_range)
    return df_reindexed

# -------------------------
# 3. Feature engineering for imputation
# -------------------------
def add_basic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day_of_week, and month as features."""
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["year"] = df.index.year
    return df

# -------------------------
# 3. Random Forest imputation
# -------------------------
def impute_noise_with_rf(df: pd.DataFrame, target_col: str = "Noise_db"):
    """
    Impute missing values in the noise column using Random Forest.
    Returns the DataFrame and the model performance (MSE).
    """
    # Train and missing splits
    train_data = df.dropna(subset=[target_col])
    missing_data = df[df[target_col].isna()]

    X_train = train_data[["hour", "day_of_week", "month"]]
    y_train = train_data[target_col]
    X_missing = missing_data[["hour", "day_of_week", "month"]]

    # Train model
    model = RandomForestRegressor(random_state=23, n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluate model
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=23
    )
    model.fit(X_train_split, y_train_split)
    y_pred = model.predict(X_test_split)
    mse = mean_squared_error(y_test_split, y_pred)

    # Predict missing values
    if not X_missing.empty:
        predicted_values = model.predict(X_missing)
        df.loc[df[target_col].isna(), target_col] = predicted_values

    return df, mse


# -------------------------
# 4. Fill static metadata columns
# -------------------------
def fill_static_columns(df: pd.DataFrame, cols: list):
    """
    Fill NaNs in static columns (like latitude, longitude, station ID)
    with the first available value.
    """
    for col in cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].dropna().iloc[0])
    return df

# -------------------------
# 5.1 Preprocessing single sensor pipeline
# -------------------------
def preprocess_single_sensor(df: pd.DataFrame, sensor_id: str) -> pd.DataFrame:
    """
    Run preprocessing only for one sensor_id.
    """
    df_sensor = df[df["Id_Instal"] == sensor_id].copy()
    
    df_sensor = initial_preprocess(df_sensor)
    df_sensor = reindex_with_full_range(df_sensor, freq="h")
    df_sensor = add_basic_time_features(df_sensor)
    df_sensor, mse = impute_noise_with_rf(df_sensor, target_col="Noise_db")
    df_sensor = fill_static_columns(df_sensor, ["Latitud", "Longitud", "Id_Instal"])

    print(f"Sensor {sensor_id} → MSE: {mse:.4f}")
    return df_sensor

# -------------------------
# 5.1 Preprocessing all sensors pipeline
# -------------------------
def preprocess_all_sensors(df: pd.DataFrame, output_dir: str = "data/interim/"):
    """
    Run preprocessing for all sensors in the dataset, one by one.
    Saves each sensor’s result to a separate parquet file.
    """
    sensor_ids = df["Id_Instal"].unique()

    for sensor_id in sensor_ids:
        df_clean = preprocess_single_sensor(df, sensor_id)
        out_path = Path(output_dir) / f"noise_clean_sensor_{sensor_id}.parquet"
        df_clean.to_parquet(out_path)
        print(f"Saved → {out_path}")


# -------------------------
# Example standalone run
# -------------------------
if __name__ == "__main__":
    # Example usage if running as a script

    sensor_id = 496  # Example sensor ID to preprocess

    df_measures_raw = load_raw_data(MEASURES_RAW_FOLDER)
    print("Raw measures data loaded.")
    
    df_sensor_raw = load_raw_data(SENSORS_RAW_FOLDER)
    print("Raw sensors data loaded.")
    
    df_raw = merge_datasets(df_measures_raw, df_sensor_raw)
    print("Datasets merged.")

    df_clean = preprocess_single_sensor(df_raw, sensor_id=sensor_id)
    print("Data preprocessing complete.")
    print(df_clean.head())
    print(df_clean.tail())
    print(df_clean.info())

    df_clean.to_parquet(INTERIM_DIR / f"sensor_{sensor_id}.parquet")
    print(f"Interim dataset saved at {INTERIM_DIR}/{sensor_id}.parquet")