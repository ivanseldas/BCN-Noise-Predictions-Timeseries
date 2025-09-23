import argparse, yaml, importlib, pandas as pd, numpy as np, matplotlib.pyplot as plt, mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
    X = df.drop(columns=["Noise_db", "Id_Instal"])
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

def log_metrics_and_plots(y_true, y_pred, model_name):
    # --- Metrics ---
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = symmetric_mape(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("smape", smape)
    mlflow.log_metric("mase", mase)

    print(f"[{model_name}] MAE={mae:.3f} | RMSE={rmse:.3f} | MAPE={mape:.2f}% | sMAPE={smape:.2f}% | MASE={mase:.3f}")

    # --- Forecast vs Actual Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Actual", linewidth=2)
    plt.plot(y_pred, label="Forecast", linewidth=2)
    plt.title(f"Forecast vs Actual - {model_name}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("forecast_vs_actual.png")
    mlflow.log_artifact("forecast_vs_actual.png")
    plt.close()

    # --- Residuals Plot ---
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 5))
    plt.plot(residuals, label="Residuals", color="red")
    plt.axhline(0, color="black", linestyle="--")
    plt.title(f"Residuals - {model_name}")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("residuals.png")
    mlflow.log_artifact("residuals.png")
    plt.close()
    
def log_predictions(y_true, y_pred, model_name):
    """Save actual vs predicted values to a CSV and log as MLflow artifact."""
    df_preds = pd.DataFrame({
        "actual": y_true,
        f"pred_{model_name}": y_pred
    })
    csv_path = f"predictions_{model_name}.csv"
    df_preds.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)

# ---------- Main ----------
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
        # Train function must return: model, y_pred
        model, y_pred = module.train(X, y, params)

        # Log parameters
        mlflow.log_params(params)

        # Log metrics and plots
        log_metrics_and_plots(y_true=y, y_pred=y_pred, model_name=args.model)
        
        # Log predictions
        log_predictions(y_true=y, y_pred=y_pred, model_name=args.model)

        # Log model
        mlflow.sklearn.log_model(model, args.model)

        print(f"✅ {args.model} trained and logged to MLflow")
