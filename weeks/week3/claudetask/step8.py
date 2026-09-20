import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#parameters
y0 = np.array([100, 0])
V1 = 10
tolerances = [1e-3, 1e-5, 1e-7, 1e-9]
time_points = [10, 100, 1000, 10000]
results = []



#normal two compartment ode
def two_compartment_ode(t, y, CL, Q, V1, V2):
    A1, A2 = y
    dA1_dt = -CL/V1 * A1 - Q * A1 / V1 + Q * A2 / V2
    dA2_dt = Q * A1 / V1 - Q * A2 / V2
    return np.array([dA1_dt, dA2_dt])

#normal solution
for tol in tolerances:
    for n_points in time_points:
        t_eval = np.linspace(0, 48, n_points)
        solution = solve_ivp(
            two_compartment_ode,
            [0,48],
            y0,
            args = (5, 2, V1, 20 ),
            atol = tol,
            rtol = tol,
            t_eval = t_eval,
        )
        A1 = solution.y[0]
        C1 = A1/V1
        Cmax = np.max(C1)
        AUC = np.trapezoid(C1, t_eval)
        results.append((tol, n_points, Cmax, AUC))

print(results)






