from fastapi import FastAPI
from api.routes import predict_now, predict

app = FastAPI(title="Noise Forecasting App")

# Include the prediction router
app.include_router(predict_now.router)
# app.include_router(predict.router)