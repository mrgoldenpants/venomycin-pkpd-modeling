import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#parameters
y0 = np.array([100, 0])
V1 = 2
V2 = 100
CL = 15
Q = 20

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
    args=[CL, Q, V1, V2],
)
#results
A1 = solution.y[0]
A2 = solution.y[1]
C1 = A1/V1
assert solution.success
assert np.min(C1) >= 0

