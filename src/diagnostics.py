"""Model-fit error metrics and model-comparison criteria."""
"""
diagnostics.py evaluates model-fit quality after parameter estimation. It
calculates residual-based metrics such as SSE, RMSE, MAE, R-squared, mean
residual, AIC, and BIC for model comparison.
"""
import numpy as np


def _errors(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.shape != predicted.shape or observed.size == 0:
        raise ValueError("observed and predicted must have the same nonzero shape.")
    return observed - predicted


def absolute_error(observed, predicted):
    return np.abs(_errors(observed, predicted))


def relative_error(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    if np.any(observed == 0):
        raise ValueError("Relative error is undefined when an observation is zero.")
    return _errors(observed, predicted) / observed


def calculate_sse(observed, predicted) -> float:
    errors = _errors(observed, predicted)
    return float(np.sum(errors**2))


def calculate_rmse(observed, predicted) -> float:
    errors = _errors(observed, predicted)
    return float(np.sqrt(np.mean(errors**2)))


def calculate_mae(observed, predicted) -> float:
    return float(np.mean(np.abs(_errors(observed, predicted))))


def calculate_r_squared(observed, predicted) -> float:
    observed = np.asarray(observed, dtype=float)
    sse = calculate_sse(observed, predicted)
    total = np.sum((observed - np.mean(observed)) ** 2)
    if total == 0:
        raise ValueError("R-squared is undefined when all observations are equal.")
    return float(1 - sse / total)


def calculate_mean_residual(observed, predicted) -> float:
    return float(np.mean(_errors(observed, predicted)))


def calculate_aic(n_observations: int, sse: float, n_parameters: int) -> float:
    if n_observations <= 0 or n_parameters < 0 or sse < 0:
        raise ValueError("Invalid AIC inputs.")
    if sse == 0:
        return float("-inf")
    return float(n_observations * np.log(sse / n_observations) + 2 * n_parameters)


def calculate_bic(n_observations: int, sse: float, n_parameters: int) -> float:
    if n_observations <= 0 or n_parameters < 0 or sse < 0:
        raise ValueError("Invalid BIC inputs.")
    if sse == 0:
        return float("-inf")
    return float(
        n_observations * np.log(sse / n_observations)
        + n_parameters * np.log(n_observations)
    )


def goodness_of_fit(observed, predicted):
    """Return the main diagnostic metrics in one dictionary."""
    return {
        "SSE": calculate_sse(observed, predicted),
        "RMSE": calculate_rmse(observed, predicted),
        "MAE": calculate_mae(observed, predicted),
        "R_squared": calculate_r_squared(observed, predicted),
        "mean_residual": calculate_mean_residual(observed, predicted),
    }
