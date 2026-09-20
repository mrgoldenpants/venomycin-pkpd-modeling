import numpy as np
from scipy.integrate import solve_ivp
V1 = 10
V2 = 20
CL = 0
Q = 2
t = [0, 48]

def two_compartment_ode(t, y, CL, Q, V1, V2):
    A1, A2 = y
    dA1_dt = -CL/V1 * A1 - Q * A1 / V1 + Q * A2 / V2
    dA2_dt = Q * A1 / V1 - Q * A2 / V2
    return np.array([dA1_dt, dA2_dt])

bolus_solution = solve_ivp(
    two_compartment_ode,
    t,
    [100, 0],
    args = (CL, 2, 10, 20 ),
)

A1 = bolus_solution.y[0]
A2 = bolus_solution.y[1]
assert A1[0] == 100
assert A1[1] + A2[1] == 100
assert np.allclose(A1 + A2, 100)


