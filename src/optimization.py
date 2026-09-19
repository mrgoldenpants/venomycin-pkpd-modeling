"""Illustrative dose-and-interval optimization functions."""
"""
optimization.py contains the illustrative dosing-optimization workflow. It
calculates regimen AUC24, checks exposure constraints, optimizes dose and dosing
interval, and independently validates an optimized repeated-dose regimen.
"""
import numpy as np
from scipy.optimize import minimize

from .metrics import calculate_auc_mic
from .models import PKPDParams
from .simulation import simulate_repeated_regimen


def regimen_auc24(dose: float, tau: float, CL: float) -> float:
    """Return the linear steady-state AUC24 approximation."""
    if dose <= 0 or tau <= 0 or CL <= 0:
        raise ValueError("dose, tau, and CL must be greater than zero.")
    return float((dose / CL) * (24.0 / tau))


def dosing_objective(x, params=PKPDParams(), target_auc24=500.0) -> float:
    """Return absolute deviation from the illustrative AUC24 target."""
    dose, tau = x
    return abs(regimen_auc24(dose, tau, params.CL) - target_auc24)


def check_auc24_lower(auc24, params=PKPDParams()) -> bool:
    return bool(auc24 >= params.AUC24_lower)


def check_auc24_upper(auc24, params=PKPDParams()) -> bool:
    return bool(auc24 <= params.AUC24_upper)


def is_regimen_feasible(dose, tau, params=PKPDParams()) -> bool:
    auc24 = regimen_auc24(dose, tau, params.CL)
    return check_auc24_lower(auc24, params) and check_auc24_upper(auc24, params)


def _lower_constraint(x, params):
    dose, tau = x
    return regimen_auc24(dose, tau, params.CL) - params.AUC24_lower


def _upper_constraint(x, params):
    dose, tau = x
    return params.AUC24_upper - regimen_auc24(dose, tau, params.CL)


def optimize_regimen(
    start,
    params=PKPDParams(),
    target_auc24=500.0,
):
    """Optimize dose and interval subject to bounds and AUC24 constraints."""
    result = minimize(
        dosing_objective,
        np.asarray(start, dtype=float),
        args=(params, target_auc24),
        method="SLSQP",
        bounds=[
            (params.Dmin, params.Dmax),
            (params.tau_min, params.tau_max),
        ],
        constraints=[
            {"type": "ineq", "fun": _lower_constraint, "args": (params,)},
            {"type": "ineq", "fun": _upper_constraint, "args": (params,)},
        ],
    )
    if not result.success:
        raise RuntimeError(f"Regimen optimization failed: {result.message}")
    return result


def optimize_from_starts(
    starts=((600.0, 20.0), (1800.0, 8.0)),
    params=PKPDParams(),
    target_auc24=500.0,
):
    """Run the optimizer from multiple starting points."""
    return [
        optimize_regimen(start, params, target_auc24)
        for start in starts
    ]


def validate_regimen(
    dose,
    tau,
    params=PKPDParams(),
    total_time=48.0,
):
    """Independently simulate a proposed regimen and report its metrics."""
    simulation = simulate_repeated_regimen(
        dose, tau, params, total_time=total_time
    )
    return {
        **simulation,
        "dose": float(dose),
        "tau": float(tau),
        "analytic_auc24": regimen_auc24(dose, tau, params.CL),
        "simulated_auc24_mic": calculate_auc_mic(
            simulation["auc24"], params.MIC
        ),
    }
