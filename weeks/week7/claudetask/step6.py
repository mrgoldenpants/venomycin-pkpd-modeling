import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
initial_guess = [4, 4, 4, 4]
one_initial_guess = [4,4]
df = pd.read_csv("/Users/goldenpants/PycharmProjects/PythonProject/pk_dataset.csv")
observed_time = df["Time"].values
observed_concentration = df["C"].values
lower_bounds = [0.05, 0.01, 3, 3]
upper_bounds = [15, 15, 20, 30]
one_lower_bounds = [0.05, 3]
one_upper_bounds = [15, 20]
tolerance = .1
A0 = 100
plot_time = np.linspace(0, 48, 100)
y0 = np.array([100, 0])

def one_compartment_ode(t, y, CL, V1):
    A1 = y[0]
    dA1_dt = -CL / V1 * A1
    return np.array([dA1_dt])


def two_compartment_ode(t, y, CL, Q, V1, V2):
    A1, A2 = y
    dA1_dt = -CL / V1 * A1 - Q * A1 / V1 + Q * A2 / V2
    dA2_dt = Q * A1 / V1 - Q * A2 / V2
    return np.array([dA1_dt, dA2_dt])


def residuals(theta, observed_time, observed_concentration):
    CL, Q, V1, V2 = theta
    y0 = np.array([100, 0])

    predicted_solution = solve_ivp(
        two_compartment_ode,
        [0, 48],
        y0,
        args=(CL, Q, V1, V2),
        t_eval=observed_time
    )

    A1 = predicted_solution.y[0]
    predicted_concentration = A1 / V1
    return observed_concentration - predicted_concentration

def one_compartment_residual(theta, observed_time, observed_concentration):
    CL, V1 = theta
    solution = solve_ivp(
        one_compartment_ode,
        t_span=[0, 48],
        y0=[A0],
        args=(CL, V1),
        t_eval= observed_time
    )

    A1 = solution.y[0]
    predicted_concentration = A1 / V1
    return observed_concentration - predicted_concentration

one_result = least_squares(one_compartment_residual, one_initial_guess, args=(observed_time, observed_concentration), bounds = (one_lower_bounds, one_upper_bounds))
two_result = least_squares(residuals, initial_guess, args=(observed_time, observed_concentration), bounds = (lower_bounds, upper_bounds))
one_CL, one_V1 = one_result.x
two_CL, two_Q, two_V1, two_V2 = two_result.x
one_prediction = solve_ivp(
    one_compartment_ode,
    [0, 48],
    y0=[A0],
    args=(one_CL, one_V1),
    t_eval=plot_time
)
A1_one = one_prediction.y[0]
one_C1 = A1_one/ one_V1

two_prediction = solve_ivp(
    two_compartment_ode,
    [0, 48],
    y0,
    args = (two_CL, two_V1, two_Q, two_V2),
    t_eval=plot_time
)
A2_two = two_prediction.y[0]
two_C2 = A2_two/ two_V1


n = len(observed_concentration)
one_k = 2
two_k = 4
one_result.fun
two_result.fun
one_error = one_result.fun
one_SSE = np.sum(one_error ** 2)
one_AIC = n*np.log(one_SSE/n) +2*one_k
one_BIC = n*np.log(one_SSE/n) + one_k * np.log(n)
two_error = two_result.fun
two_SSE = np.sum(two_error ** 2)
two_AIC = n*np.log(two_SSE/n) +2*two_k
two_BIC = n*np.log(two_SSE/n) + two_k * np.log(n)

comparison_table = pd.DataFrame({
    "Model": ["One Compartment", "Two Compartment"],
    "AIC": [one_AIC, two_AIC],
    "BIC": [one_BIC, two_BIC]
})

print(comparison_table)
