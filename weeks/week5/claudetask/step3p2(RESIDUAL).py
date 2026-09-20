import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def residuals(theta):
    CL, Q, V1, V2 = theta
    y0 = np.array([100, 0])
    observed_time = np.linspace(0, 48, 100)

    def two_compartment_ode(t, y, CL, Q, V1, V2):
        A1, A2 = y
        dA1_dt = -CL / V1 * A1 - Q * A1 / V1 + Q * A2 / V2
        dA2_dt = Q * A1 / V1 - Q * A2 / V2
        return np.array([dA1_dt, dA2_dt])
    true_solution = solve_ivp(
        two_compartment_ode,
        [0, 48],
        y0,
        args=(5, 2, 10, 20),
        t_eval=observed_time
    )
    observed_concentration = true_solution.y[0]/10
    predicted_solution = solve_ivp(
        two_compartment_ode,
        [0, 48],
        y0,
        args=(CL, Q, V1, V2),
        t_eval = observed_time
    )

    A1 = predicted_solution.y[0]
    predicted_concentration = A1/V1
    return observed_concentration - predicted_concentration

def objective(theta):
    residual = residuals(theta)
    return np.sum(residual**2)

assert objective([5, 2, 10, 20]) >= 0