# src/models/sarimax.py
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")


def train(X, y, params):
    """
    Train a SARIMAX model with given params.
    Casts y and X to float and uses safe param keys for MLflow logging.
    """
    # Endogenous: y | Exogenous: X
    y = y.astype(float)
    exog = X.astype(float) if X.shape[1] > 0 else None

    # Usar los valores para el modelo
    model = SARIMAX(
        y,
        exog=exog,
        order=(
            params.get("order_p", 1),
            params.get("order_d", 0),
            params.get("order_q", 1)
        ),
        seasonal_order=(
            params.get("seasonal_P", 0),
            params.get("seasonal_D", 0),
            params.get("seasonal_Q", 0),
            params.get("seasonal_s", 24)
        ),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)

    preds = results.predict(start=0, end=len(y)-1, exog=exog)

    # Devolver modelo + métricas
    return results, preds


def objective(trial, X, y):
    """
    Optuna objective for SARIMAX tuning.
    """
    y = y.astype(float)
    exog = X.astype(float) if X.shape[1] > 0 else None

    params = {
        "order_p": trial.suggest_int("order_p", 0, 3),
        "order_d": trial.suggest_int("order_d", 0, 2),
        "order_q": trial.suggest_int("order_q", 0, 3),
        "seasonal_P": trial.suggest_int("seasonal_P", 0, 2),
        "seasonal_D": trial.suggest_int("seasonal_D", 0, 1),
        "seasonal_Q": trial.suggest_int("seasonal_Q", 0, 2),
        "seasonal_s": trial.suggest_categorical("seasonal_s", [0, 12, 24, 168]),
    }

    try:
        model = SARIMAX(
            y,
            exog=exog,
            order=(params["order_p"], params["order_d"], params["order_q"]),
            seasonal_order=(params["seasonal_P"], params["seasonal_D"], params["seasonal_Q"], params["seasonal_s"]),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = model.fit(disp=False)
        preds = results.predict(start=0, end=len(y)-1, exog=exog)
        rmse = mean_squared_error(y, preds)
        return rmse
    except Exception:
        return np.inf