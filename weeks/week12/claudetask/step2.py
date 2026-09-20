"""
NOTE: ILLUSTRATIVE MODEL FOR DEMONSTRATION PURPOSES ONLY

This module simulates a 2-compartment PK model linked to an Emax PD response.
It serves as an educational tool to demonstrate numerical integration,
exposure metrics, and saturation kinetics.

CLINICAL DISCLAIMER:
- This script does NOT represent true bedside clinical targets for drugs like vancomycin.
- True clinical targets rely on steady-state AUC24/MIC ratios (e.g., 400–600).
- The current model calculates single-dose AUC over 0-48 hours (AUC_0_48).
- AUC_0_48 in this demonstration should not be interpreted as steady-state AUC24.
- Do not use this model or its equations for clinical dosing decisions.
"""

from typing import Dict, NamedTuple, Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
#--------
# 1. PARAMETER CONTAINER AND ODE DEFINITION
#--------
class PKPDParams(NamedTuple):
    #Pharmacokinetics
    CL: float = 5.0 #Clearance (L/h)
    Q: float = 2.0 #Intercompartmental Clearance (L/h)
    V1: float = 10.0 #Central Volume (L)
    V2: float = 20.0 #Peripheral Volume (L)
    Dmin: float = 500 #Minimum dosage (mg)
    Dmax: float = 2000 #Maximum dosage (mg)
    tau_min: float = 6 #Minimum dosing interval (h)
    tau_max: float = 24 #Maximum dosing interval (h)
    #Pharmacodynamics
    MIC: float = 2.0 #Minimum Inhibitory Concentration (mg/L)
    E0: float = 1.0 #Baseline Effect
    Emax: float = 5.0 #Maximum Drug Effect
    AUC_MIC50: float = 10.0 #AUC/MIC ratio producing 50% of Emax

def two_compartment_ode(
    t: float, y: np.ndarray, CL: float, Q: float, V1: float, V2: float
) -> np.ndarray:
    """Calculates rate of change for central (A1) and peripheral (A2) amounts."""
    A1, A2 = y
    dA1_dt = -(CL / V1) * A1 - (Q / V1) * A1 + (Q / V2) * A2
    dA2_dt = (Q / V1) * A1 - (Q / V2) * A2
    return np.array([dA1_dt, dA2_dt])

#--------
#2. CORE PK/PD CALCULATIONS
#--------
def calculate_auc_mic(auc: float, mic: float) -> float:
    """Calculates the exposure-to-susceptibility ratio (AUC/MIC)."""
    if mic <= 0:
        raise ValueError("MIC must be strictly greater than zero.")
    return auc / mic


def auc_mic_response(
    emax: float, e0: float, auc_mic50: float, current_ratio: float
) -> float:
    """Calculates PD response using an Emax saturation model."""
    return e0 + (emax * current_ratio) / (auc_mic50 + current_ratio)

#--------
#3. PIPELINE EXECUTION
#--------
def run_pipeline(
    dose: float,
    params: PKPDParams = PKPDParams(),
    t_span: Tuple[float, float] = (0.0, 48.0),
    n_points: int = 1000,
) -> Dict[str, float]:
    """Runs a single dose through the full simulation pipeline."""
    y0 = np.array([dose, 0.0])
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    solution = solve_ivp(
        two_compartment_ode,
        t_span,
        y0,
        args=(params.CL, params.Q, params.V1, params.V2),
        t_eval=t_eval,
        method="RK45",
    )
    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")

    A1 = solution.y[0]
    C1 = A1 / params.V1

    # Numerical integration across dense grid
    auc_0_48 = np.trapezoid(C1, solution.t)
    auc_mic = calculate_auc_mic(auc_0_48, params.MIC)
    effect = auc_mic_response(
        params.Emax, params.E0, params.AUC_MIC50, auc_mic
    )

    return {
        "dose": dose,
        "auc_0_48": auc_0_48,
        "auc_mic": auc_mic,
        "effect": effect,
    }
#--------
#4. VERIFICATIONS AND ASSERTIONS
#--------
if __name__ == "__main__":
    default_params = PKPDParams()

    res_100 = run_pipeline(100.0, default_params)
    res_200 = run_pipeline(200.0, default_params)

    print(
        f"Dose 100 mg -> AUC/MIC: {res_100['auc_mic']:.2f} | Effect: {res_100['effect']:.4f}"
    )
    print(
        f"Dose 200 mg -> AUC/MIC: {res_200['auc_mic']:.2f} | Effect: {res_200['effect']:.4f}"
    )

    # Sanity checks
    assert (
        res_200["effect"] > res_100["effect"]
    ), "Monotonicity failure: Higher dose must produce greater effect."
    assert (
        res_200["auc_0_48"] > res_100["auc_0_48"]
    ), "PK failure: Higher dose must yield higher exposure."
    print("\n✓ Pipeline verification checks passed successfully.")
    # Dosing-bound check
    assert not (
            default_params.tau_min <= 0.5 <= default_params.tau_max
    ), "Bounds failure: 0.5 h should not be an allowed dosing interval."

#--------
#4. TEST AND ADDONS
#--------
auc_24 = 500
auc_target = 500
def objective(auc_24, auc_target):
    return abs(auc_24 - auc_target)

assert objective(auc_24, auc_target) == 0
