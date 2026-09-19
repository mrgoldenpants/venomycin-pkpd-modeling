"""Numerical simulation functions for the PK/PD pipeline."""
"""
simulation.py runs numerical PK simulations using the model equations. It
simulates single-dose one- and two-compartment profiles, repeated-dose regimens,
and the complete dose-to-exposure-to-effect pipeline.
"""

from typing import Dict, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .metrics import (
    calculate_auc,
    calculate_auc24,
    calculate_auc_mic,
    calculate_cmax,
    calculate_tmax,
    calculate_trough,
)
from .models import (
    PKPDParams,
    auc_mic_response,
    one_compartment_ode,
    two_compartment_ode,
)


def simulate_one_compartment(
    dose: float,
    CL: float,
    V1: float,
    t_span: Tuple[float, float] = (0.0, 48.0),
    n_points: int = 1000,
) -> Dict[str, np.ndarray]:
    """Simulate a single IV bolus in a one-compartment model."""
    time = np.linspace(t_span[0], t_span[1], n_points)
    solution = solve_ivp(
        one_compartment_ode,
        t_span,
        [dose],
        args=(CL, V1),
        t_eval=time,
        method="RK45",
    )
    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")
    amount = solution.y[0]
    return {
        "time": solution.t,
        "amount": amount,
        "concentration": amount / V1,
    }


def simulate_two_compartment(
    dose: float,
    params: PKPDParams = PKPDParams(),
    t_span: Tuple[float, float] = (0.0, 48.0),
    n_points: int = 1000,
) -> Dict[str, np.ndarray]:
    """Simulate a single central-compartment IV bolus."""
    if dose < 0:
        raise ValueError("dose cannot be negative.")
    time = np.linspace(t_span[0], t_span[1], n_points)
    solution = solve_ivp(
        two_compartment_ode,
        t_span,
        [dose, 0.0],
        args=(params.CL, params.Q, params.V1, params.V2),
        t_eval=time,
        method="RK45",
    )
    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")
    return {
        "time": solution.t,
        "A1": solution.y[0],
        "A2": solution.y[1],
        "concentration": solution.y[0] / params.V1,
    }


def run_pipeline(
    dose: float,
    params: PKPDParams = PKPDParams(),
    t_span: Tuple[float, float] = (0.0, 48.0),
    n_points: int = 1000,
) -> Dict[str, object]:
    """Run a single dose through PK simulation, exposure, and illustrative PD."""
    simulation = simulate_two_compartment(dose, params, t_span, n_points)
    time = simulation["time"]
    concentration = simulation["concentration"]
    auc_total = calculate_auc(time, concentration)
    auc_24 = calculate_auc24(time, concentration, start_time=t_span[0])
    auc_mic = calculate_auc_mic(auc_total, params.MIC)
    effect = auc_mic_response(
        params.Emax, params.E0, params.AUC_MIC50, auc_mic
    )
    return {
        "dose": dose,
        "time": time,
        "A1": simulation["A1"],
        "A2": simulation["A2"],
        "concentration": concentration,
        "auc_0_48": auc_total,
        "auc_0_24": auc_24,
        "auc_mic": auc_mic,
        "cmax": calculate_cmax(concentration),
        "tmax": calculate_tmax(time, concentration),
        "trough": calculate_trough(concentration),
        "effect": effect,
    }


def simulate_repeated_regimen(
    dose: float,
    tau: float,
    params: PKPDParams = PKPDParams(),
    total_time: float = 48.0,
    n_points_per_interval: int = 100,
) -> Dict[str, object]:
    """Simulate repeated IV boluses while carrying state between intervals."""
    if dose <= 0 or tau <= 0 or total_time <= 0:
        raise ValueError("dose, tau, and total_time must be greater than zero.")

    state = np.array([0.0, 0.0])
    all_times = []
    all_concentrations = []
    dose_times = np.arange(0.0, total_time, tau)

    for index, dose_time in enumerate(dose_times):
        state = state.copy()
        state[0] += dose
        end_time = (
            dose_times[index + 1]
            if index < len(dose_times) - 1
            else total_time
        )
        interval_time = np.linspace(dose_time, end_time, n_points_per_interval)
        solution = solve_ivp(
            two_compartment_ode,
            (dose_time, end_time),
            state,
            args=(params.CL, params.Q, params.V1, params.V2),
            t_eval=interval_time,
            method="RK45",
        )
        if not solution.success:
            raise RuntimeError(f"ODE solver failed: {solution.message}")
        all_times.extend(solution.t)
        all_concentrations.extend(solution.y[0] / params.V1)
        state = solution.y[:, -1]

    time = np.asarray(all_times)
    concentration = np.asarray(all_concentrations)
    start_24 = max(0.0, total_time - 24.0)
    auc_total = calculate_auc(time, concentration)
    auc_24 = calculate_auc24(time, concentration, start_time=start_24)
    return {
        "time": time,
        "concentration": concentration,
        "auc": auc_total,
        "auc24": auc_24,
        "auc_mic": calculate_auc_mic(auc_24, params.MIC),
        "cmax": calculate_cmax(concentration),
        "trough": calculate_trough(concentration),
        "dose_times": dose_times,
    }
