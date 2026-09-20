import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
initial_guess = [1, 0.5, 8]


def residuals(theta):
    CL, Q, V1 = theta
    V2 = 15
    y0 = np.array([100, 0])
    observed_time = np.array([0, 10, 20, 30, 40])

    def two_compartment_ode(t, y, CL, Q, V1, V2):
        A1, A2 = y
        dA1_dt = -CL / V1 * A1 - Q * A1 / V1 + Q * A2 / V2
        dA2_dt = Q * A1 / V1 - Q * A2 / V2
        return np.array([dA1_dt, dA2_dt])

    observed_concentration = np.array([10.0, 7.52, 5.65, 4.25, 3.19])
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

result = least_squares(residuals, initial_guess, bounds = ([0.05, 0, 3], [15, 15, 20]))

assert np.all(np.abs(residuals(result.x)) < 0.05)