import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
y0 = np.array([100, 0])
t_eval_10 = np.linspace(0, 48, 10)
t_eval_10000 = np.linspace(0, 48, 10000)



#normal twocompartmentode below
def two_compartment_ode(t, y, CL, Q, V1, V2):
    A1, A2 = y
    dA1_dt = -CL/V1 * A1 - Q * A1 / V1 + Q * A2 / V2
    dA2_dt = Q * A1 / V1 - Q * A2 / V2
    return np.array([dA1_dt, dA2_dt])
#this is the solution for when we measure t for 10 different points
solution_10 = solve_ivp(
        two_compartment_ode,
        [0,48],
        y0,
        args = (5, 2, 10, 20 ),
        rtol = 1e-4,
        atol = 1e-4,
        t_eval = t_eval_10,
)
#this is the solution for when we measure t for 10000 different points
solution_10000 = solve_ivp(
        two_compartment_ode,
        [0,48],
        y0,
        args = (5, 2, 10, 20 ),
        rtol = 1e-4,
        atol = 1e-4,
        t_eval = t_eval_10000,
)
#we use allclose for arrays
assert np.allclose(solution_10.y[:, -1], solution_10000.y[:, -1])



