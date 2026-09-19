"""Reusable PK exposure and summary metrics."""
"""
metrics.py calculates reusable PK exposure and summary metrics from
concentration-time data, including AUC, AUC24, Cmax, Tmax, trough concentration,
half-life, and AUC/MIC.
"""
import numpy as np


def _curve_arrays(time, concentration):
    time = np.asarray(time, dtype=float)
    concentration = np.asarray(concentration, dtype=float)
    if time.ndim != 1 or concentration.ndim != 1:
        raise ValueError("time and concentration must be one-dimensional.")
    if time.size != concentration.size or time.size == 0:
        raise ValueError("time and concentration must have the same nonzero length.")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(concentration)):
        raise ValueError("time and concentration must contain finite values.")
    if np.any(np.diff(time) < 0):
        raise ValueError("time must be sorted in nondecreasing order.")
    return time, concentration


def calculate_auc(time, concentration) -> float:
    """Calculate numerical AUC using the trapezoidal rule."""
    time, concentration = _curve_arrays(time, concentration)
    return float(np.trapezoid(concentration, time))


def trapezoid_auc(time, concentration) -> float:
    """Calculate AUC manually using consecutive trapezoids."""
    time, concentration = _curve_arrays(time, concentration)
    total_auc = 0.0
    for index in range(len(time) - 1):
        width = time[index + 1] - time[index]
        average_height = (concentration[index] + concentration[index + 1]) / 2
        total_auc += width * average_height
    return float(total_auc)


def calculate_auc_window(time, concentration, start_time, duration=24.0) -> float:
    """Calculate AUC within an exact time window, interpolating its boundaries."""
    time, concentration = _curve_arrays(time, concentration)
    end_time = start_time + duration
    if duration <= 0:
        raise ValueError("duration must be greater than zero.")
    if start_time < time[0] or end_time > time[-1]:
        raise ValueError("The requested AUC window lies outside the simulated curve.")

    # Keep the last concentration at repeated bolus timestamps for interpolation.
    reversed_time = time[::-1]
    _, reversed_indices = np.unique(reversed_time, return_index=True)
    keep = np.sort(time.size - 1 - reversed_indices)
    unique_time = time[keep]
    unique_concentration = concentration[keep]

    inside = (unique_time > start_time) & (unique_time < end_time)
    window_time = np.concatenate(([start_time], unique_time[inside], [end_time]))
    window_concentration = np.concatenate(
        (
            [np.interp(start_time, unique_time, unique_concentration)],
            unique_concentration[inside],
            [np.interp(end_time, unique_time, unique_concentration)],
        )
    )
    return calculate_auc(window_time, window_concentration)


def calculate_auc24(time, concentration, start_time=0.0) -> float:
    """Calculate AUC over a 24-hour window."""
    return calculate_auc_window(time, concentration, start_time, duration=24.0)


def calculate_cmax(concentration) -> float:
    """Return the maximum observed or simulated concentration."""
    concentration = np.asarray(concentration, dtype=float)
    if concentration.size == 0:
        raise ValueError("concentration cannot be empty.")
    return float(np.max(concentration))


def calculate_tmax(time, concentration) -> float:
    """Return the time at which the maximum concentration first occurs."""
    time, concentration = _curve_arrays(time, concentration)
    return float(time[np.argmax(concentration)])


def calculate_trough(concentration) -> float:
    """Return the final concentration in a dosing interval or simulation."""
    concentration = np.asarray(concentration, dtype=float)
    if concentration.size == 0:
        raise ValueError("concentration cannot be empty.")
    return float(concentration[-1])


def calculate_half_life(ke=None, CL=None, V=None) -> float:
    """Calculate half-life from ke, or from CL and V when ke is omitted."""
    if ke is None:
        if CL is None or V is None:
            raise ValueError("Provide ke, or provide both CL and V.")
        ke = CL / V
    if ke <= 0:
        raise ValueError("ke must be strictly greater than zero.")
    return float(np.log(2) / ke)


def calculate_auc_mic(auc: float, mic: float) -> float:
    """Calculate the exposure-to-susceptibility ratio AUC/MIC."""
    if mic <= 0:
        raise ValueError("MIC must be strictly greater than zero.")
    return float(auc / mic)
