import mlflow

MODEL_NAME = "NoiseForecasting"
MODEL_ALIAS = "production"
EXPORT_PATH = "models/production/"

if __name__ == "__main__":
    print(f"📦 Exporting {MODEL_NAME}@{MODEL_ALIAS} to {EXPORT_PATH} ...")

    # Download model artifacts from MLflow Registry
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}",
        dst_path=EXPORT_PATH
    )

    print(f"✅ Model exported successfully to {EXPORT_PATH}")
