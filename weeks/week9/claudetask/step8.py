import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
MIC = 1
#parameters
y0 = np.array([100, 0])
V1 = 10
Emax = 5
E0 = 1
AUC_MIC50 = 10

#normal two compartment ode
def two_compartment_ode(t, y, CL, Q, V1, V2):
    A1, A2 = y
    dA1_dt = -CL/V1 * A1 - Q * A1 / V1 + Q * A2 / V2
    dA2_dt = Q * A1 / V1 - Q * A2 / V2
    return np.array([dA1_dt, dA2_dt])

#normal solution

solution = solve_ivp(
        two_compartment_ode,
        [0,48],
        y0,
        args = (5, 2, V1, 20 ),
)
#results
A1 = solution.y[0]
C1 = A1/V1
t = solution.t
AUC = np.trapezoid(C1, t)

def calculate_auc_mic(AUC, MIC):
    if MIC <= 0:
        raise ValueError("MIC must be greater than zero")
    return AUC/MIC

auc_mic = calculate_auc_mic(AUC, MIC)

def auc_mic_response(Emax, E0, AUC_MIC50, auc_mic):
    E = E0 + (Emax * auc_mic) / (AUC_MIC50 + auc_mic)
    return E

E = auc_mic_response(Emax, E0, AUC_MIC50, auc_mic)

print("AUC:", AUC)
print("AUC/MIC:", auc_mic)
print("Final Emax Response", E)