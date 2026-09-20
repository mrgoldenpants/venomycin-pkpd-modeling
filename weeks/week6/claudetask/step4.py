import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares



initial_guess = [4, 4, 4]
df = pd.read_csv("pk_dataset_3.csv")
sparse_df = df[df["Time"] % 20 == 0]
def residuals(theta, observed_time, observed_concentration):
    CL, Q, V1 = theta
    V2 = 15
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
        result = least_squares(residuals, initial_guess, args=(observed_time, observed_concentration))
        fitted_parameters.append([
            subject,
            result.x[0],
            result.x[1],
            result.x[2]
        ])

    results_df = pd.DataFrame(
        fitted_parameters,
        columns=["Subject", "CL", "Q", "V1"]
    )




    return results_df


import matplotlib.pyplot as plt
fitted_parameters = fit_model(df)

subject = 1
subject_data = df[df["Subjects"] == subject]

observed_time = subject_data["Time"].values
observed_concentration = subject_data["C"].values
subject_fit = fitted_parameters[fitted_parameters["Subject"] == subject]

CL = subject_fit["CL"].values[0]
Q = subject_fit["Q"].values[0]
V1 = subject_fit["V1"].values[0]

V2 = 15
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

A1 = solution.y[0]

fitted_concentration = A1 / V1

plt.plot(observed_time, observed_concentration, "o", label="Observed")
plt.plot(observed_time, fitted_concentration, "-", label="Fitted")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.legend()
plt.show()
plt.semilogy(observed_time, observed_concentration, "o", label="Observed")
plt.semilogy(observed_time, fitted_concentration, "-", label="Fitted")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.legend()
plt.show()


