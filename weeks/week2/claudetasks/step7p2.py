import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


V1 = 10
V2 = 20
CL = 5
Q = 2
t = [0, 48]
R = 15
def two_compartment_ode(t, y, CL, Q, V1, V2):
    A1, A2 = y
    dA1_dt = -CL/V1 * A1 - Q * A1 / V1 + Q * A2 / V2
    dA2_dt = Q * A1 / V1 - Q * A2 / V2
    return np.array([dA1_dt, dA2_dt])

def two_compartment_infusion(t, y, CL, Q, V1, V2, R):
    A1, A2 = y
    dA1_dt=  R - CL/V1 * A1 - Q * A1 / V1 + Q * A2 / V2
    dA2_dt = Q * A1 / V1 - Q * A2 / V2
    return np.array([dA1_dt, dA2_dt])

bolus_solution = solve_ivp(
    two_compartment_ode,
    t,
    [100, 0],
    args = (5, 2, 10, 20 ),
)
infusion_solution = solve_ivp(
    two_compartment_infusion,
    t,
    [0, 0],
    args = (5, 2, 10, 20, R),

)

A1 = bolus_solution.y[0]
A2 = bolus_solution.y[1]
A1_infusion = infusion_solution.y[0]
A2_infusion = infusion_solution.y[1]
#how do i get the first value of A1
assert A1[0] == 100
assert A1_infusion[0] == 0

assert R * (t[1] -t[0]) == 720