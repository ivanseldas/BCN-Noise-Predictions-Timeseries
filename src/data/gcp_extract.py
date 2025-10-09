"""
gcp_extract.py
---------------------------------
Module to query and retrieve noise sensor data from Google BigQuery.
Generates hourly averaged noise levels per sensor for GIS visualization.
"""

from google.cloud import bigquery
import pandas as pd
from pathlib import Path


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
        l.Id_Instal,
        ANY_VALUE(l.Longitud) AS Longitud,
        ANY_VALUE(l.Latitud) AS Latitud,
        AVG(m.Nivell_LAeq_1h) AS Avg_Noise_dB,
        MIN(m.Nivell_LAeq_1h) AS Min_Noise_dB,
        MAX(m.Nivell_LAeq_1h) AS Max_Noise_dB
    FROM `noise-forecasting-bcn-472319.sensors.sensor_location` AS l
    RIGHT JOIN `noise-forecasting-bcn-472319.bcnnoise.noise_measures` AS m
    ON m.Id_Instal = l.Id_Instal
    WHERE l.Id_Instal IS NOT NULL
    GROUP BY l.Id_Instal
    ORDER BY l.Id_Instal;
    """

    print("🚀 Running query on BigQuery...")
    df = client.query(query).to_dataframe()
    df.to_csv(output_path, index=False)
    print(f"✅ Sensor summary dataset saved: {output_path}")

    return df


if __name__ == "__main__":
    # Example manual run
    fetch_hourly_avg(
        project_id="noise-forecasting-bcn-472319",
        dataset_table="bcnnoise.noise_measures",
        output_path="data/processed/sensor_summary.csv"
    )