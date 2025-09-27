import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

class SarimaxWrapper:
    def __init__(self, params):
        self.params = params
        self.model_ = None
        self.results_ = None
        self.train_len_ = None

    def fit(self, X, y):
        y = y.astype(float)
        exog = X.astype(float) if X.shape[1] > 0 else None

        self.train_len_ = len(y)

        self.model_ = SARIMAX(
            y,
            exog=exog,
            order=(
                self.params.get("order_p", 1),
                self.params.get("order_d", 0),
                self.params.get("order_q", 1),
            ),
            seasonal_order=(
                self.params.get("seasonal_P", 0),
                self.params.get("seasonal_D", 0),
                self.params.get("seasonal_Q", 0),
                self.params.get("seasonal_s", 24),
            ),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.results_ = self.model_.fit(disp=False)
        return self

    def predict(self, X_test):
        # start/end en base al train_len_
        start = self.train_len_
        end = self.train_len_ + len(X_test) - 1
        exog = X_test.astype(float) if X_test.shape[1] > 0 else None
        return self.results_.predict(start=start, end=end, exog=exog)


# -------------------------
# For Training
# -------------------------
def train(X, y, params):
    model = SarimaxWrapper(params)
    return model.fit(X, y)


# -------------------------
# For Hyperparameter Tuning with Optuna
# -------------------------
def objective(trial, X, y):
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
