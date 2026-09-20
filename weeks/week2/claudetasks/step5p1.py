import numpy as np
from scipy.integrate import solve_ivp


def two_compartment_ode(t, y, CL, Q, V1, V2):
    A1, A2 = y
    dA1_dt = -CL/V1 * A1 - Q * A1 / V1 + Q * A2 / V2
    dA2_dt = Q * A1 / V1 - Q * A2 / V2
    return np.array([dA1_dt, dA2_dt])

solution = solve_ivp(
    two_compartment_ode,
    [0, 48],
    [100, 0],
    args = (5, 2, 10, 20 ),
)

A1 = solution.y[0]
A2 = solution.y[1]

