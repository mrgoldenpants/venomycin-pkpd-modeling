"""Mathematical PK and illustrative PD model definitions."""
"""
models.py contains the mathematical definitions of the PK/PD system. It defines
the PKPDParams parameter container, the one- and two-compartment ODE equations,
and the illustrative Emax pharmacodynamic response model.
"""

from typing import NamedTuple

import numpy as np


class PKPDParams(NamedTuple):
    """Parameters used by the illustrative two-compartment PK/PD model."""

    CL: float = 5.0
    Q: float = 2.0
    V1: float = 10.0
    V2: float = 20.0
    Dmin: float = 500.0
    Dmax: float = 2000.0
    tau_min: float = 6.0
    tau_max: float = 24.0
    MIC: float = 2.0
    E0: float = 1.0
    Emax: float = 5.0
    AUC_MIC50: float = 10.0
    AUC24_lower: float = 400.0
    AUC24_upper: float = 600.0


def validate_pk_parameters(
    CL: float,
    Q: float,
    V1: float,
    V2: float,
) -> None:
    """Raise informative errors for invalid PK parameter values."""
    if CL < 0:
        raise ValueError("CL cannot be negative.")
    if Q < 0:
        raise ValueError("Q cannot be negative.")
    if V1 <= 0:
        raise ValueError("V1 must be strictly greater than zero.")
    if V2 <= 0:
        raise ValueError("V2 must be strictly greater than zero.")

def one_compartment_ode(
    t: float, y: np.ndarray, CL: float, V1: float
) -> np.ndarray:
    """Return the rate of change of amount in a one-compartment model."""
    validate_pk_parameters(CL, 0.0, V1, 1.0)
    del t
    A1 = y[0]
    return np.array([-(CL / V1) * A1])


def two_compartment_ode(
    t: float,
    y: np.ndarray,
    CL: float,
    Q: float,
    V1: float,
    V2: float,
) -> np.ndarray:
    """Return central and peripheral amount derivatives."""
    validate_pk_parameters(CL, Q, V1, V2)
    del t
    A1, A2 = y
    dA1_dt = -(CL / V1) * A1 - (Q / V1) * A1 + (Q / V2) * A2
    dA2_dt = (Q / V1) * A1 - (Q / V2) * A2
    return np.array([dA1_dt, dA2_dt])


def auc_mic_response(
    emax: float,
    e0: float,
    auc_mic50: float,
    current_ratio: float,
) -> float:
    """Return an illustrative saturating Emax response to AUC/MIC."""
    if auc_mic50 <= 0:
        raise ValueError("AUC_MIC50 must be strictly greater than zero.")
    if current_ratio < 0:
        raise ValueError("AUC/MIC cannot be negative.")
    return e0 + (emax * current_ratio) / (auc_mic50 + current_ratio)
