import numpy as np
from scipy.integrate import solve_ivp
tolerances = [1e-3, 1e-5, 1e-7]
V1 = 10
y0 = np.array([100, 0])
Cmax_values = []




def two_compartment_ode(t, y, CL, Q, V1, V2):
    A1, A2 = y
    dA1_dt = -CL/V1 * A1 - Q * A1 / V1 + Q * A2 / V2
    dA2_dt = Q * A1 / V1 - Q * A2 / V2
    return np.array([dA1_dt, dA2_dt])
for tol in tolerances:
    solution = solve_ivp(
        two_compartment_ode,
        [0,48],
        y0,
        args = (5, 2, 10, 20 ),
        rtol = tol,
        atol = tol,

)
    A1 = solution.y[0]
    C1 = A1 / V1
    Cmax = np.max(C1)

    Cmax_values.append(Cmax)
assert np.isclose(Cmax_values[-2], Cmax_values[-1])


