import numpy as np
from scipy.integrate import solve_ivp


Dose = 100
CL = 5
Q = 2
V1 = 10
V2 = 20

analytical_AUC = Dose / CL

def two_compartment_ode(t, y, CL, Q, V1, V2):
    A1, A2 = y
    dA1_dt = -CL/V1 * A1 - Q * A1 / V1 + Q * A2 / V2
    dA2_dt = Q * A1 / V1 - Q * A2 / V2
    return np.array([dA1_dt, dA2_dt])

def trapezoid_auc(t, c):
    total_auc = 0.0
    for i in range(len(t) - 1):
        time_width = t[i + 1] - t[i]
        average = (c[i] + c[i + 1]) / 2
        area = time_width * average
        total_auc += area

    return total_auc
solution = solve_ivp(
    two_compartment_ode,
    [0, 500],
    [Dose, 0],
    args=(CL, Q, V1, V2),
    t_eval=np.linspace(0, 500, 5000)
)
A1 = solution.y[0]
C1 = A1 / V1
numerical_AUC = trapezoid_auc(solution.t, C1)

percent_difference = ((np.abs(numerical_AUC - analytical_AUC))/analytical_AUC) * 100
print("Analytical AUC: ", analytical_AUC)
print("Numerical AUC: ", numerical_AUC)
print("Percentage difference: ", percent_difference)
assert np.isclose(numerical_AUC, analytical_AUC, rtol=1e-3)