# src/validation/validate_data.py
import pandas as pd
import great_expectations as ge
from src.utils.paths import PROCESSED_DIR


def validate_data(input_path: str):
    """
    Validate the feature-engineered dataset using Great Expectations.
    """

    # -------------------------
    # 1. Load dataset
    # -------------------------
    df = pd.read_parquet(input_path)
    df_ge = ge.dataset.PandasDataset(df)

    # -------------------------
    # 2. Expectations
    # -------------------------

    # Columns should exist
    expected_columns = [
        "Id_Instal", "Noise_db", "hour", "day_of_week", "month", "year",
        "Latitud", "Longitud", "is_weekend",
        "Lockdown_During lockdown", "Lockdown_Post-lockdown", "Lockdown_Pre-lockdown",
        "public_holiday",
        "Noise_db_lag_1", "Noise_db_lag_24",
        "Noise_db_roll_mean_3", "Noise_db_roll_std_3",
        "Noise_db_roll_mean_24", "Noise_db_roll_std_24",
        "sin_1", "cos_1", "sin_2", "cos_2"
    ]
    df_ge.expect_table_columns_to_match_set(expected_columns)

    # Noise should be non-null and within a plausible range
    df_ge.expect_column_values_to_not_be_null("Noise_db")
    df_ge.expect_column_values_to_be_between("Noise_db", min_value=30, max_value=120)

    # Time-based features
    df_ge.expect_column_values_to_be_between("hour", 0, 23)
    df_ge.expect_column_values_to_be_between("day_of_week", 0, 6)
    df_ge.expect_column_values_to_be_between("month", 1, 12)
    df_ge.expect_column_values_to_be_between("year", 2015, 2030)

    # Lat/Long should be in Barcelona bounds
    df_ge.expect_column_values_to_be_between("Latitud", 41.2, 41.6)
    df_ge.expect_column_values_to_be_between("Longitud", 2.0, 2.3)

    # Categorical encodings must be binary (0/1 or True/False)
    for col in ["is_weekend", "Lockdown_During lockdown", "Lockdown_Post-lockdown",
                "Lockdown_Pre-lockdown", "public_holiday"]:
        df_ge.expect_column_values_to_be_in_set(col, [0, 1, True, False])

    # Rolling features can have NaNs at the beginning but should be numeric otherwise
    for col in ["Noise_db_lag_1", "Noise_db_lag_24",
                "Noise_db_roll_mean_3", "Noise_db_roll_std_3",
                "Noise_db_roll_mean_24", "Noise_db_roll_std_24"]:
        df_ge.expect_column_values_to_be_of_type(col, "float")

    # Fourier features must be between -1 and 1
    for col in ["sin_1", "cos_1", "sin_2", "cos_2"]:
        df_ge.expect_column_values_to_be_between(col, -1.0, 1.0)

    # -------------------------
    # 3. Validate
    # -------------------------
    results = df_ge.validate()
    print(results)

    # Fail if validation unsuccessful
    if not results["success"]:
        raise ValueError("Data validation failed. See details above.")

    print("✅ Data validation passed successfully.")


if __name__ == "__main__":
    processed_path = PROCESSED_DIR / "sensor_496_processed.parquet"
    validate_data(processed_path)