from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# -------------------------
# For Training
# -------------------------
def train(X, y, params):
    model = GradientBoostingRegressor(**params)
    model.fit(X, y)
    return model
# -------------------------
# For Hyperparameter Tuning with Optuna
# -------------------------
def objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, step=0.01),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
    }
    model = GradientBoostingRegressor(**params, random_state=42)
    tscv = TimeSeriesSplit(n_splits=3)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_root_mean_squared_error")
    return -scores.mean()