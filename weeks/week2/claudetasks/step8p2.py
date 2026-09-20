import numpy as np


t = [0, 48]

parameter_sets = [
    (10, 20, 5, 2),
    (5, 10, 2, 1),
    (20, 40, 10, 4)
]
def two_compartment_ode(t, y, CL, Q, V1, V2):
    A1, A2 = y
    dA1_dt = -CL/V1 * A1 - Q * A1 / V1 + Q * A2 / V2
    dA2_dt = Q * A1 / V1 - Q * A2 / V2
    return np.array([dA1_dt, dA2_dt])
for V1, V2, CL, Q in parameter_sets:
    bolus_solution = solve_ivp(
        two_compartment_ode,
        t,
        [100, 0],
        args = (CL, 2, 10, 20 ),
)

A1 = bolus_solution.y[0]
A2 = bolus_solution.y[1]
C1 = A1/V1
assert np.all(C1 >= 0)
