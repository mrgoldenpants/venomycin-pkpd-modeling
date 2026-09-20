import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
initial_guess = [4, 4, 4, 4]
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
        observed_time = subject_data["Time"].values
        observed_concentration = subject_data["C"].values
        result = least_squares(residuals, initial_guess, args=(observed_time, observed_concentration),bounds = (lower_bounds, upper_bounds))
        errors = result.fun
        predicted_concentration = observed_concentration - errors
        plt.scatter(observed_time, errors)
        plt.xlabel("Time")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Time")
        plt.axhline(0, linestyle="--")
        plt.show()
        plt.scatter(predicted_concentration, errors)
        plt.xlabel("Predicted Concentration")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Predicted Concentration")
        plt.axhline(0, linestyle="--")
        plt.show()
        plt.scatter(observed_concentration, errors)
        plt.xlabel("Observation Concentration")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Observation Concentration")
        plt.axhline(0, linestyle="--")
        plt.show()

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

fit_model(df)
