import argparse, yaml, importlib, pandas as pd, numpy as np, mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from src.utils.paths import PROCESSED_DIR, CONFIGS_DIR

# ---------- Helpers ----------
def load_config(path):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return None

def load_data():
    df = pd.read_parquet(PROCESSED_DIR / "sensor_496_processed.parquet")
    y = df["Noise_db"].values
    X = df.drop(columns=["Noise_db", "Id_Instal"]).values
    return X, y

# --- Forecasting Metrics ---
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100

def symmetric_mape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def mean_absolute_scaled_error(y_true, y_pred):
    naive_forecast = y_true[:-1]
    mae_naive = mean_absolute_error(y_true[1:], naive_forecast)
    mae_model = mean_absolute_error(y_true, y_pred)
    return mae_model / mae_naive if mae_naive != 0 else np.inf

def compute_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "smape": symmetric_mape(y_true, y_pred),
        "mase": mean_absolute_scaled_error(y_true, y_pred),
    }

# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (gradient_boost, xgboost, prophet, lstm, sarimax)")
    parser.add_argument("--mode", required=True, choices=["cv", "production"], help="Training mode: cv or production")
    args = parser.parse_args()

    X, y = load_data()
    cfg = load_config(CONFIGS_DIR / f"model_{args.model}.yaml")
    params = cfg["model"] if cfg else {}
    module = importlib.import_module(f"src.models.{args.model}")

    mlflow.set_experiment("Noise Forecasting")

    # ----------------- CV Mode -----------------
    if args.mode == "cv":
        with mlflow.start_run(run_name=f"{args.model}_cv", nested=False):
            mlflow.log_params(params)

            tscv = TimeSeriesSplit(n_splits=3)
            all_metrics = []

            for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
                with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    model = module.train(X_train, y_train, params)
                    y_pred = model.predict(X_test)

                    metrics = compute_metrics(y_test, y_pred)
                    mlflow.log_metrics({f"{k}": v for k, v in metrics.items()})

                    all_metrics.append(metrics)

                    print(
                        f"[Fold {fold}] "
                        + " | ".join([f"{k.upper()}={v:.3f}" for k, v in metrics.items()])
                    )

            # Compute averages across folds
            avg_metrics = {f"cv_{k}": np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
            mlflow.log_metrics(avg_metrics)

            print(
                f"✅ CV completed for {args.model}: "
                + " | ".join([f"{k.upper()}={v:.3f}" for k, v in avg_metrics.items()])
            )

    # ----------------- Production Mode -----------------
    elif args.mode == "production":
        with mlflow.start_run(run_name=f"{args.model}_production", nested=False):
            mlflow.log_params(params)

            # Train on the full dataset
            model = module.train(X, y, params)

            # Log final model
            mlflow.sklearn.log_model(sk_model=model, name=args.model)

            # Mark run as production
            mlflow.set_tag("stage", "production")

            print(f"✅ {args.model} trained on full dataset and logged to MLflow")
