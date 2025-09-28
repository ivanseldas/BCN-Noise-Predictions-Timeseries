# src/config/paths.py
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SENSORS_RAW_FOLDER = RAW_DIR / "sensors"
MEASURES_RAW_FOLDER = RAW_DIR / "noise_measures"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"


# Model paths
MODEL_DIR = ROOT_DIR / "models"
PRODUCTION_DIR = MODEL_DIR / "production"

# Config paths
CONFIGS_DIR = ROOT_DIR / "src/configs"
RANDOM_FOREST_CONFIG = CONFIGS_DIR / "model_random_forest.yaml"
XGBOOST_CONFIG = CONFIGS_DIR / "model_xgboost.yaml"
LSTM_CONFIG = CONFIGS_DIR / "model_lstm.yaml"