# src/train.py
import argparse, yaml, importlib, pandas as pd, numpy as np, mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from src.utils.paths import INTERIM_DIR, PROCESSED_DIR, CONFIGS_DIR
from src.features.feature_engineering import build_features_with_hash
from src.evaluation.metrics import compute_metrics

# ---------- Helpers ----------
def load_config(path):
    """Load YAML configuration file."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return None


def load_interim_data():
    """Load raw interim dataset before feature engineering."""
    return pd.read_parquet(INTERIM_DIR / "sensor_496.parquet")


# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Model name (gradient_boost, xgboost, prophet, lstm, sarimax)")
    parser.add_argument("--mode", required=True, choices=["cv", "production"],
                        help="Training mode: cv or production")
    args = parser.parse_args()

    # 1. Load config
    cfg = load_config(CONFIGS_DIR / f"model_{args.model}.yaml")
    params = cfg["model"] if cfg else {}

    # 2. Load interim dataset
    df = load_interim_data()

    # 3. Build features + feature hash
    df_features, feature_hash, feature_config = build_features_with_hash(df)
    df_features.dropna(inplace=True)

    # 4. Prepare X, y
    y = df_features["Noise_db"].values
    X = df_features.drop(columns=["Noise_db", "Id_Instal"]).values

    # 5. Import model
    module = importlib.import_module(f"src.models.{args.model}")

    # 6. Start MLflow experiment
    mlflow.set_experiment("Noise Forecasting")

    # ----------------- CV Mode -----------------
    if args.mode == "cv":
        with mlflow.start_run(run_name=f"{args.model}_cv", nested=False):
            # Log params + feature info
            mlflow.log_params(params)
            mlflow.log_param("feature_hash", feature_hash)
            mlflow.log_dict(feature_config, "feature_config.json")

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

                    print(f"[Fold {fold}] " + " | ".join(
                        [f"{k.upper()}={v:.3f}" for k, v in metrics.items()])
                    )

            # Log averages
            avg_metrics = {f"{k}": np.mean([m[k] for m in all_metrics])
                           for k in all_metrics[0]}
            mlflow.log_metrics(avg_metrics)

            print(f"✅ CV completed for {args.model}: " +
                  " | ".join([f"{k.upper()}={v:.3f}" for k, v in avg_metrics.items()]))

    # ----------------- Production Mode -----------------
    elif args.mode == "production":
        with mlflow.start_run(run_name=f"{args.model}_production", nested=False):
            # Log params + feature info
            mlflow.log_params(params)
            mlflow.log_param("feature_hash", feature_hash)
            mlflow.log_dict(feature_config, "feature_config.json")

            # Train on full dataset
            model = module.train(X, y, params)

            # Save processed dataset for traceability
            save_path = PROCESSED_DIR / f"sensor_496_features_{feature_hash}.parquet"
            df_features.to_parquet(save_path)
            mlflow.log_artifact(save_path, artifact_path="features")

            # Log final model
            mlflow.sklearn.log_model(sk_model=model, name=args.model)

            mlflow.set_tag("stage", "production")
            print(f"✅ {args.model} trained and logged with feature hash {feature_hash}")
