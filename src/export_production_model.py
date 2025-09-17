import os
import joblib
import mlflow

MODEL_NAME = "NoiseForecasting"
MODEL_ALIAS = "production"
EXPORT_DIR = "models/production"
OUTPUT_PKL_PATH = os.path.join(EXPORT_DIR, "model.pkl")

if __name__ == "__main__":
    print(f"📦 Exporting {MODEL_NAME}@{MODEL_ALIAS} to {OUTPUT_PKL_PATH} ...")
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # Load model from MLflow Model Registry
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

    # Save as pure pickle
    joblib.dump(model, OUTPUT_PKL_PATH)

    print(f"✅ Model exported successfully as pure pickle to {OUTPUT_PKL_PATH}")
