import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
fitted_parameters = []
initial_guess = [4, 4, 4]
df = pd.read_csv("pk_dataset_3.csv")
subjects = df["Subjects"].unique()


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
for subject in subjects:
    subject_data = df[df["Subjects"] == subject]
    observed_time = subject_data["Time"].values
    observed_concentration = subject_data["C"].values
    result = least_squares(residuals, initial_guess, args=(observed_time, observed_concentration))
    fitted_parameters.append([
        subject,
        result.x[0],
        result.x[1],
        result.x[2]
    ])



import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
fitted_parameters = []
initial_guess = [4, 4, 4]
df = pd.read_csv("pk_dataset_3.csv")
subjects = df["Subjects"].unique()


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
for subject in subjects:
    subject_data = df[df["Subjects"] == subject]
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

print(results_df)

assert len(result.x) == 3