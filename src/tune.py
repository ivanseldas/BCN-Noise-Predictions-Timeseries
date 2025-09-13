import argparse, yaml, importlib, pandas as pd, optuna
from src.utils.paths import PROCESSED_DIR, CONFIGS_DIR

# -------------------------
# Load config and data
# -------------------------
def load_data():
    df = pd.read_parquet(PROCESSED_DIR / "sensor_496_processed.parquet")
    y = df["Noise_db"]
    X = df.drop(columns=["Noise_db", "Id_Instal"])
    return X, y

# -------------------------
# Save best hyperparameters to config
# -------------------------
def save_best(params, model_name):
    path = CONFIGS_DIR / f"model_{model_name}.yaml"
    try:
        with open(path, "r") as f: cfg = yaml.safe_load(f)
    except FileNotFoundError:
        cfg = {"model": {}}
    cfg["model"].update(params)
    with open(path, "w") as f: yaml.dump(cfg, f)

# -------------------------
# Main execution for hyperparameter tuning
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add model and trials arguments
    parser.add_argument("--model", required=True, help="Model name (random_forest, xgboost, prophet, lstm)")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials")
    args = parser.parse_args()

    X, y = load_data()
    # Dynamically import the model module (e.g., random_forest.py)
    module = importlib.import_module(f"src.models.{args.model}")

    # Define the Optuna study and optimize
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: module.objective(trial, X, y), n_trials=args.trials)

    # Get the best hyperparameters and score
    best_params = study.best_params
    best_score = study.best_value

    # Save the best hyperparameters to the config file
    save_best(best_params, args.model)
    print(f"✅ Best params for {args.model}: {best_params} | Best CV RMSE: {best_score:.3f}")