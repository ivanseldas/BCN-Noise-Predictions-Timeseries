import argparse, yaml, importlib, pandas as pd, mlflow
from src.utils.paths import PROCESSED_DIR, CONFIGS_DIR

def load_config(path):
    try:
        with open(path, "r") as f: return yaml.safe_load(f)
    except FileNotFoundError:
        return None

def load_data():
    df = pd.read_parquet(PROCESSED_DIR / "sensor_496_processed.parquet")
    y = df["Noise_db"]
    X = df.drop(columns=["Noise_db", "Id_Instal"])
    return X, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (random_forest, xgboost, prophet, lstm, sarimax)")
    args = parser.parse_args()

    X, y = load_data()
    cfg = load_config(CONFIGS_DIR / f"model_{args.model}.yaml")

    params = cfg["model"] if cfg else {}
    module = importlib.import_module(f"src.models.{args.model}")

    mlflow.set_experiment("Noise Forecasting")
    with mlflow.start_run(run_name=f"{args.model}_train", nested=False):
        model, rmse = module.train(X, y, params)
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, args.model)
        print(f"✅ {args.model} trained | RMSE: {rmse:.3f}")
