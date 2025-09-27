import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# -------------------------
# For Training
# -------------------------
def train(X, y, params):
    model = xgb.XGBRegressor(
        **params,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

# -------------------------
# For Hyperparameter Tuning with Optuna
# -------------------------
def objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, step=0.01),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.1),
        "gamma": trial.suggest_float("gamma", 0, 5, step=0.5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }

    model = xgb.XGBRegressor(
        **params,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror"
    )

    tscv = TimeSeriesSplit(n_splits=3)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_root_mean_squared_error")
    return -scores.mean()