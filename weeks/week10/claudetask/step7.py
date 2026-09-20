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
from SALib.sample import sobol
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
#--------
# 1. PARAMETER CONTAINER AND ODE DEFINITION
#--------
class PKPDParams(NamedTuple):
    #Pharmacokinetics
    CL: float = 5.0 #Clearance (L/h)
    Q: float = 2.0 #Intercompartmental Clearance (L/h)
    V1: float = 10.0 #Central Volume (L)
    V2: float = 20.0 #Peripheral Volume (L)
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
problem = {
    "num_vars": 4,
    "names": ["CL", "Q", "V1", "V2"],
    "bounds": [
        [0.5, 25.0],    # CL
        [2.0, 30.0],    # Q
        [3.0, 8.0],     # V1
        [15.0, 150.0]   # V2
    ]
}
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
    cmax = np.max(C1)

    # Trough at 24 hours
    trough = np.interp(24.0, solution.t, C1)

    # AUC from 0-24 hours
    mask_24 = solution.t <= 24.0
    auc_0_24 = np.trapezoid(
        C1[mask_24],
        solution.t[mask_24]
    )

    # Numerical integration across dense grid
    auc_0_48 = np.trapezoid(C1, solution.t)
    auc_mic = calculate_auc_mic(auc_0_48, params.MIC)
    effect = auc_mic_response(
        params.Emax, params.E0, params.AUC_MIC50, auc_mic
    )

    return {
        "dose": dose,
        "auc_0_48": auc_0_48,
        "auc_0_24": auc_0_24,
        "auc_mic": auc_mic,
        "cmax": cmax,
        "trough": trough,
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

#--------
#5. TEST
#--------
N = 1024

param_values = sobol.sample(problem, N, calc_second_order=True)
param_values_original = param_values.copy()
auc_results = []
auc24_results = []
cmax_results = []
trough_results = []
for sample in param_values:
    CL, Q, V1, V2 = sample
    sample_params = PKPDParams(
        CL = CL,
        Q = Q,
        V1 = V1,
        V2 = V2)
    result = run_pipeline(100, sample_params)
    #results
    auc_results.append(result["auc_0_48"])
    auc24_results.append(result["auc_0_24"])
    cmax_results.append(result["cmax"])
    trough_results.append(result["trough"])

auc_results = np.array(auc_results)
auc24_results = np.array(auc24_results)
cmax_results = np.array(cmax_results)
trough_results = np.array(trough_results)
print(auc_results.shape)
print(auc_results[:5])
Si_auc24 = sobol_analyze.analyze(
    problem,
    auc24_results,
    calc_second_order=True,
    print_to_console=False
)

Si_cmax = sobol_analyze.analyze(
    problem,
    cmax_results,
    calc_second_order=True,
    print_to_console=False
)

Si_trough = sobol_analyze.analyze(
    problem,
    trough_results,
    calc_second_order=True,
    print_to_console=False
)
assert np.array_equal(param_values, param_values_original)
assert len(auc24_results) == len(param_values)
assert len(cmax_results) == len(param_values)
assert len(trough_results) == len(param_values)
print("\nAUC24 Sobol Results:")
for name, s1, st in zip(problem["names"], Si_auc24["S1"], Si_auc24["ST"]):
    print(f"{name}: S1 = {s1:.4f}, ST = {st:.4f}")

print("\nCmax Sobol Results:")
for name, s1, st in zip(problem["names"], Si_cmax["S1"], Si_cmax["ST"]):
    print(f"{name}: S1 = {s1:.4f}, ST = {st:.4f}")

print("\nTrough Sobol Results:")
for name, s1, st in zip(problem["names"], Si_trough["S1"], Si_trough["ST"]):
    print(f"{name}: S1 = {s1:.4f}, ST = {st:.4f}")