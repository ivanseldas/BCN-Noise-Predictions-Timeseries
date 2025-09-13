from pathlib import Path
from src.config.paths import DATA_DIR, RAW_DIR, PROCESSED_DIR, SENSORS_RAW_FOLDER, MEASURES_RAW_FOLDER
from src.data.load_data import load_raw_data, merge_datasets


df_sensors = load_raw_data(folder_path=SENSORS_RAW_FOLDER)
df_noise = load_raw_data(folder_path=MEASURES_RAW_FOLDER)

df = merge_datasets(df_noise, df_sensors)

print(df.columns)