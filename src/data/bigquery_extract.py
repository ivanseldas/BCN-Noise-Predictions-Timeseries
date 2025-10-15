"""
gcp_extract.py
---------------------------------
Module to query and retrieve noise sensor data from Google BigQuery.
Generates hourly averaged noise levels per sensor for GIS visualization.
"""

from google.cloud import bigquery
import pandas as pd
from pathlib import Path

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure credentials are visible to Google libraries
if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    print(f"✅ Using credentials from: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
else:
    raise EnvironmentError("❌ GOOGLE_APPLICATION_CREDENTIALS not set in .env file.")

def fetch_hourly_avg(project_id: str, dataset_table: str, output_path: str) -> pd.DataFrame:
    """
    Fetch hourly average noise per sensor from BigQuery and export to CSV.

    Parameters
    ----------
    project_id : str
        GCP project ID (e.g. 'noise-forecasting-bcn-472319')
    dataset_table : str
        Full BigQuery table path (e.g. 'bcnnoise.noise_measures')
    output_path : str
        Local path to save the CSV (e.g. 'data/processed/noise_hourly_avg.csv')

    Returns
    -------
    pd.DataFrame
        Hourly aggregated noise dataset.
    """
    client = bigquery.Client(project=project_id)

    query = f"""
    SELECT
    CASE
        WHEN SAFE_CAST(REGEXP_EXTRACT(CAST(Hora AS STRING), r'^\d{{1,2}}') AS INT64) = 24 THEN
        TIMESTAMP_ADD(
            TIMESTAMP(CONCAT(
            CAST(`Any` AS STRING), '-',
            LPAD(CAST(Mes AS STRING), 2, '0'), '-',
            LPAD(CAST(Dia AS STRING), 2, '0'), ' 00:00:00'
            )),
            INTERVAL 1 DAY
        )
        ELSE TIMESTAMP(CONCAT(
        CAST(`Any` AS STRING), '-',
        LPAD(CAST(Mes AS STRING), 2, '0'), '-',
        LPAD(CAST(Dia AS STRING), 2, '0'), ' ',
        LPAD(
            CAST(SAFE_CAST(REGEXP_EXTRACT(CAST(Hora AS STRING), r'^\d{{1,2}}') AS INT64) AS STRING),
            2, '0'
        ),
        ':00:00'
        ))
    END AS datetime_hour,
    Id_Instal AS sensor_id,
    ROUND(AVG(SAFE_CAST(Nivell_LAeq_1h AS FLOAT64)), 1) AS avg_noise_level,
    COUNT(Nivell_LAeq_1h) AS num_measurements
    FROM `{dataset_table}`
    WHERE SAFE_CAST(`Any` AS INT64) BETWEEN 2015 AND 2025
    GROUP BY datetime_hour, sensor_id
    ORDER BY datetime_hour, sensor_id;

    """

    print("🚀 Running query on BigQuery...")
    df = client.query(query).to_dataframe()
    df.to_parquet(output_path, index=False, compression='snappy')

    print(f"✅ Sensor summary dataset saved: {output_path}")

    return df


if __name__ == "__main__":
    # Example manual run
    fetch_hourly_avg(
        project_id="noise-forecasting-bcn-472319",
        dataset_table="bcnnoise.noise_measures",
        output_path="data/interim/noise_hourly_avg.parquet"
    )