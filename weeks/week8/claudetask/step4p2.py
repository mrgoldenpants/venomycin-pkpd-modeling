import numpy as np
from scipy.integrate import solve_ivp
upper_bounds = [12, 24, 48]
auc_bounds = []

#parameters
y0 = np.array([100, 0])
V1 = 10

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
A2 = solution.y[1]
C1 = A1/V1
total_mass = A1 + A2

def trapezoid_auc(t, c):
    total_auc = 0.0
    for i in range(len(t) - 1):
        time_width = t[i + 1] - t[i]
        average = (c[i] + c[i + 1]) / 2
        area = time_width * average
        total_auc += area

    return total_auc
AUC = trapezoid_auc(solution.t, C1)

for bound in upper_bounds:
    mask = solution.t <= bound
    time_subset = solution.t[mask]
    concentration_subset = C1[mask]
    auc = trapezoid_auc(time_subset, concentration_subset)
    auc_bounds.append(auc)
    print(auc)

assert auc_bounds[0] < auc_bounds[1] < auc_bounds[2]

