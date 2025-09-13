import os
import glob
import pandas as pd

def load_raw_data(folder_path: str, pattern: str = "*.csv") -> pd.DataFrame:
    """Load and concatenate measurement CSVs from a folder.

    Supports glob patterns like '*.csv'.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Find files by pattern
    csv_files = sorted(glob.glob(os.path.join(folder_path, pattern)))
    if not csv_files:
        raise FileNotFoundError(f"No files matching {pattern} in {folder_path}")

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Loop through the files and read them into pandas
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        dataframes.append(df)

    # Return concatenated DataFrames
    return pd.concat(dataframes, ignore_index=True)

def merge_datasets(measurements: pd.DataFrame, sensors: pd.DataFrame, on: str = "Id_Instal") -> pd.DataFrame:
    
    # Merge the datasets on the specified column
    return pd.merge(measurements, sensors, on=on, how="left")