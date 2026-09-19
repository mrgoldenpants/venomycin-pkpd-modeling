"""Parameter estimation and uncertainty functions."""
"""
fitting.py estimates PK parameters from observed concentration-time data. It
contains prediction and residual functions, weighted and unweighted least-squares
fitting, multi-subject fitting, uncertainty estimates, and bootstrap fitting.
"""
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .models import two_compartment_ode


DEFAULT_INITIAL_GUESS = np.array([4.0, 4.0, 4.0, 4.0])
DEFAULT_LOWER_BOUNDS = np.array([0.05, 0.01, 3.0, 3.0])
DEFAULT_UPPER_BOUNDS = np.array([15.0, 15.0, 20.0, 30.0])


def predict_concentration(theta, observation_time, dose=100.0):
    """Predict central concentration at arbitrary observation times."""
    CL, Q, V1, V2 = np.asarray(theta, dtype=float)
    observation_time = np.asarray(observation_time, dtype=float)
    if observation_time.ndim != 1 or observation_time.size == 0:
        raise ValueError("observation_time must be a nonempty one-dimensional array.")
    if np.any(observation_time < 0):
        raise ValueError("observation times cannot be negative.")

    unique_time = np.unique(observation_time)
    solution = solve_ivp(
        two_compartment_ode,
        (0.0, float(max(48.0, unique_time[-1]))),
        [dose, 0.0],
        args=(CL, Q, V1, V2),
        t_eval=unique_time,
        method="RK45",
    )
    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")
    unique_prediction = solution.y[0] / V1
    return np.interp(observation_time, unique_time, unique_prediction)


def residuals(theta, observed_time, observed_concentration, dose=100.0):
    """Return observed minus predicted concentrations."""
    observed = np.asarray(observed_concentration, dtype=float)
    predicted = predict_concentration(theta, observed_time, dose)
    if observed.shape != predicted.shape:
        raise ValueError("observed_time and observed_concentration must match.")
    return observed - predicted


def weighted_residuals(
    theta, observed_time, observed_concentration, dose=100.0
):
    """Return proportional residuals, weighted by 1/observed concentration."""
    observed = np.asarray(observed_concentration, dtype=float)
    if np.any(observed <= 0):
        raise ValueError("Weighted residuals require positive observations.")
    return residuals(theta, observed_time, observed, dose) / observed


def sum_squared_residuals(
    theta, observed_time, observed_concentration, dose=100.0, weighted=False
) -> float:
    """Return the sum of squared weighted or unweighted residuals."""
    function = weighted_residuals if weighted else residuals
    errors = function(theta, observed_time, observed_concentration, dose)
    return float(np.sum(errors**2))


def fit_subject(
    observed_time,
    observed_concentration,
    initial_guess=DEFAULT_INITIAL_GUESS,
    bounds=(DEFAULT_LOWER_BOUNDS, DEFAULT_UPPER_BOUNDS),
    dose=100.0,
    weighted=False,
):
    """Fit CL, Q, V1, and V2 for one subject."""
    function = weighted_residuals if weighted else residuals
    result = least_squares(
        function,
        np.asarray(initial_guess, dtype=float),
        args=(
            np.asarray(observed_time, dtype=float),
            np.asarray(observed_concentration, dtype=float),
            dose,
        ),
        bounds=bounds,
    )
    if not result.success:
        raise RuntimeError(f"Parameter fit failed: {result.message}")
    return result


def parameter_standard_errors(fit_result):
    """Estimate parameter standard errors from the least-squares Jacobian."""
    jacobian = fit_result.jac
    n_observations, n_parameters = jacobian.shape
    if n_observations <= n_parameters:
        return np.full(n_parameters, np.nan)
    residual_variance = np.sum(fit_result.fun**2) / (
        n_observations - n_parameters
    )
    covariance = residual_variance * np.linalg.pinv(jacobian.T @ jacobian)
    return np.sqrt(np.maximum(np.diag(covariance), 0.0))


def _subject_column(data):
    for name in ("Subjects", "Subject"):
        if name in data.columns:
            return name
    raise ValueError("Data must contain a 'Subjects' or 'Subject' column.")


def fit_multiple_subjects(
    data,
    initial_guess=DEFAULT_INITIAL_GUESS,
    bounds=(DEFAULT_LOWER_BOUNDS, DEFAULT_UPPER_BOUNDS),
    dose=100.0,
    weighted=False,
):
    """Fit every subject in a PK DataFrame and return one row per subject."""
    subject_column = _subject_column(data)
    rows = []
    for subject in data[subject_column].unique():
        subject_data = data[data[subject_column] == subject]
        result = fit_subject(
            subject_data["Time"].to_numpy(),
            subject_data["C"].to_numpy(),
            initial_guess,
            bounds,
            dose,
            weighted,
        )
        standard_errors = parameter_standard_errors(result)
        rows.append(
            {
                "Subject": subject,
                "CL": result.x[0],
                "Q": result.x[1],
                "V1": result.x[2],
                "V2": result.x[3],
                "CL_SE": standard_errors[0],
                "Q_SE": standard_errors[1],
                "V1_SE": standard_errors[2],
                "V2_SE": standard_errors[3],
                "cost": result.cost,
            }
        )
    return pd.DataFrame(rows)


def fit_robust(
    data,
    initial_guess=DEFAULT_INITIAL_GUESS,
    lower_bounds=DEFAULT_LOWER_BOUNDS,
    upper_bounds=DEFAULT_UPPER_BOUNDS,
    dose=100.0,
    weighted=False,
):
    """Reproducible multi-subject fitting entry point."""
    return fit_multiple_subjects(
        data,
        initial_guess=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        dose=dose,
        weighted=weighted,
    )


def bootstrap_fit(
    data,
    n_bootstrap,
    seed=42,
    initial_guess=DEFAULT_INITIAL_GUESS,
    bounds=(DEFAULT_LOWER_BOUNDS, DEFAULT_UPPER_BOUNDS),
    dose=100.0,
):
    """Resample observations within subject and refit each bootstrap dataset."""
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be greater than zero.")
    subject_column = _subject_column(data)
    rng = np.random.default_rng(seed)
    outputs = []
    for bootstrap_index in range(n_bootstrap):
        sampled_subjects = []
        for subject in data[subject_column].unique():
            subject_data = data[data[subject_column] == subject]
            indices = rng.integers(0, len(subject_data), size=len(subject_data))
            sampled_subjects.append(subject_data.iloc[indices])
        bootstrap_data = pd.concat(sampled_subjects, ignore_index=True)
        fit = fit_multiple_subjects(
            bootstrap_data,
            initial_guess=initial_guess,
            bounds=bounds,
            dose=dose,
        )
        fit.insert(0, "bootstrap", bootstrap_index)
        outputs.append(fit)
    return pd.concat(outputs, ignore_index=True)
