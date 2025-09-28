# src/evaluation/backtesting.py
import argparse
import importlib
import mlflow
import pandas as pd
from src.utils.paths import INTERIM_DIR, CONFIGS_DIR
from src.features.feature_engineering import build_features_with_hash
from src.evaluation.metrics import compute_backtest_metrics


# -------------------------
# Backtesting routine
# -------------------------
def run_backtest(model_module, X, y, params, n_splits=3, test_size=168*4):
    """
    Expanding-window backtest.
    - n_splits: number of folds
    - test_size: length of each test window (e.g. 168 = one week of hourly data)
    """
    all_metrics = []
    n_samples = len(y)
    fold_size = (n_samples - test_size) // n_splits

    for fold in range(1, n_splits + 1):
        train_end = fold_size * fold
        test_end = train_end + test_size

        X_train, X_test = X[:train_end], X[train_end:test_end]
        y_train, y_test = y[:train_end], y[train_end:test_end]

        if len(y_test) == 0:
            break

        with mlflow.start_run(run_name=f"backtest_fold_{fold}", nested=True):
            # Train model
            model = model_module.train(X_train, y_train, params)
            y_pred = model.predict(X_test)

            # Compute metrics (includes baseline + improvement)
            metrics = compute_backtest_metrics(y_train, y_test, y_pred)
            mlflow.log_metrics(metrics)

            all_metrics.append(metrics)

            print(
                f"[Backtest Fold {fold}] "
                f"RMSE={metrics['rmse']:.3f} | "
                f"Baseline RMSE={metrics['baseline_rmse']:.3f} | "
                f"Improvement={metrics['rmse_improvement_pct']:.2f}%"
            )

    # Aggregate metrics across folds
    if all_metrics:
        avg_metrics = {
            f"{k}": sum(m[k] for m in all_metrics) / len(all_metrics)
            for k in all_metrics[0]
        }
        return all_metrics, avg_metrics
    return [], {}


# -------------------------
# CLI entrypoint
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (random_forest, xgboost, sarimax, etc.)")
    parser.add_argument("--n_splits", type=int, default=3, help="Number of backtest folds")
    parser.add_argument("--test_size", type=int, default=168, help="Length of each test window")
    args = parser.parse_args()

    # Load interim dataset
    df = pd.read_parquet(INTERIM_DIR / "sensor_496.parquet")

    # Build features
    df_features, feature_hash, feature_config = build_features_with_hash(df)
    df_features.dropna(inplace=True)

    y = df_features["Noise_db"].values
    X = df_features.drop(columns=["Noise_db", "Id_Instal"]).values

    # Load model module + params
    import yaml
    cfg_path = CONFIGS_DIR / f"model_{args.model}.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    params = cfg["model"] if cfg else {}
    module = importlib.import_module(f"src.models.{args.model}")

    # Start MLflow experiment
    mlflow.set_experiment("Noise Forecasting")

    with mlflow.start_run(run_name=f"{args.model}_backtest", nested=False):
        mlflow.log_params(params)
        mlflow.log_param("feature_hash", feature_hash)
        mlflow.log_dict(feature_config, "feature_config.json")

        all_metrics, avg_metrics = run_backtest(
            module, X, y, params, n_splits=args.n_splits, test_size=args.test_size
        )

        # Log aggregate metrics at the parent run
        if avg_metrics:
            mlflow.log_metrics(avg_metrics)

    print(f"✅ Backtesting completed for {args.model} with feature hash {feature_hash}")