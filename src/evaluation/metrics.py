# src/evaluation/metrics.py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


#--- Metrics for forecasting tasks ---

def mean_absolute_percentage_error(y_true, y_pred):
    """MAPE (%)"""
    return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100


def symmetric_mape(y_true, y_pred):
    """sMAPE (%)"""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) /
                         (np.abs(y_true) + np.abs(y_pred) + 1e-8))


def mean_absolute_scaled_error(y_true, y_pred):
    """MASE (relative to naive forecast)"""
    naive_forecast = y_true[:-1]
    mae_naive = mean_absolute_error(y_true[1:], naive_forecast)
    mae_model = mean_absolute_error(y_true, y_pred)
    return mae_model / mae_naive if mae_naive != 0 else np.inf


def compute_metrics(y_true, y_pred):
    """Unified metrics dict for forecasting tasks"""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "smape": symmetric_mape(y_true, y_pred),
        "mase": mean_absolute_scaled_error(y_true, y_pred),
    }



# --- Baseline forecast and metrics ---

def naive_last_forecast(y_train, horizon):
    """Baseline forecast = repeat last observed value."""
    return np.repeat(y_train[-1], horizon)

def compute_backtest_metrics(y_train, y_test, y_pred):
    """
    Compute regular metrics + baseline comparison.
    """
    metrics = compute_metrics(y_test, y_pred)

    # Baseline (naive last)
    y_naive = naive_last_forecast(y_train, len(y_test))
    baseline_rmse = mean_squared_error(y_test, y_naive)

    # Relative improvement
    rel_improvement = (baseline_rmse - metrics["rmse"]) / baseline_rmse * 100

    # Extend metrics dict
    metrics.update({
        "baseline_rmse": baseline_rmse,
        "rmse_improvement_pct": rel_improvement
    })

    return metrics