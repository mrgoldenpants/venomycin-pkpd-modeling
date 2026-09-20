import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
initial_guess = [0.6, 0.3, 9, 14]
true_parameters = [0.5, 0.2, 10, 15]
df = pd.read_csv("/Users/goldenpants/PycharmProjects/PythonProject/pk_dataset.csv")
lower_bounds = [0.05, 0.01, 3, 3]
upper_bounds = [15, 15, 20, 30]
def residuals(theta, observed_time, observed_concentration):
    CL, Q, V1, V2 = theta
    y0 = np.array([100, 0])

    def two_compartment_ode(t, y, CL, Q, V1, V2):
        A1, A2 = y
        dA1_dt = -CL / V1 * A1 - Q * A1 / V1 + Q * A2 / V2
        dA2_dt = Q * A1 / V1 - Q * A2 / V2
        return np.array([dA1_dt, dA2_dt])

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
def fit_model(data):
    fitted_parameters = []
    subjects = data["Subjects"].unique()
    for subject in subjects:
        subject_data = data[data["Subjects"] == subject]
        observed_time = np.array([0, 10, 20, 30, 40])
        observed_concentration = generate_data(true_parameters, observed_time)
        result = least_squares(residuals, initial_guess, args=(observed_time, observed_concentration),bounds = (lower_bounds, upper_bounds))
        errors = result.fun
        SSE = np.sum(errors ** 2)
        RMSE = np.sqrt(np.mean(errors ** 2))
        MAE = np.mean(np.abs(errors))
        SS_total = np.sum(
            (observed_concentration - np.mean(observed_concentration)) ** 2
        )
        R_squared = 1 - SSE / SS_total
        print(SSE, RMSE, MAE, R_squared)
        assert SSE < 1e-6
        assert RMSE < 1e-6
        assert MAE < 1e-6
        assert abs(R_squared - 1) < 1e-6

        fitted_parameters.append([
            subject,
            result.x[0],
            result.x[1],
            result.x[2],
            result.x[3]
        ])
    results_df = pd.DataFrame(
        fitted_parameters,
        columns=["Subject", "CL", "Q", "V1", "V2"]
    )
    return results_df



def generate_data(theta, observed_time):
    CL, Q, V1, V2 = theta
    y0 = np.array([100, 0])
    def two_compartment_ode(t, y, CL, Q, V1, V2):
        A1, A2 = y
        dA1_dt = -CL / V1 * A1 - Q * A1 / V1 + Q * A2 / V2
        dA2_dt = Q * A1 / V1 - Q * A2 / V2
        return np.array([dA1_dt, dA2_dt])

    solution = solve_ivp(
        two_compartment_ode,
        [0, 48],
        y0,
        args=(CL, Q, V1, V2),
        t_eval=observed_time
    )
    return solution.y[0]/V1


fit_model(df)
