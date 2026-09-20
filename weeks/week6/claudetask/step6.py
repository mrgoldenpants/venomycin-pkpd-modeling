import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
initial_guess = [4, 4, 4, 4]
df = pd.read_csv("pk_dataset_3.csv")
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
    unique_time = np.unique(observed_time)

    predicted_solution = solve_ivp(
        two_compartment_ode,
        [0, 48],
        y0,
        args=(CL, Q, V1, V2),
        t_eval=unique_time
    )

    A1 = predicted_solution.y[0]
    predicted_concentration = A1 / V1
    predicted_at_observation = np.interp(observed_time, unique_time, predicted_concentration)
    return observed_concentration - predicted_at_observation
def fit_model(data):
    fitted_parameters = []
    subjects = data["Subjects"].unique()
    for subject in subjects:
        subject_data = data[data["Subjects"] == subject]
        observed_time = subject_data["Time"].values
        observed_concentration = subject_data["C"].values
        result = least_squares(residuals, initial_guess, args=(observed_time, observed_concentration),bounds = (lower_bounds, upper_bounds))
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

def bootstrap(data, n_bootstrap):
    bootstrap_results = []
    subjects = data["Subjects"].unique()

    for i in range(n_bootstrap):
        bootstrap_subjects = []
        for subject in subjects:
            subject_data = data[data["Subjects"] == subject]

            resample_data = subject_data.sample(n = len(subject_data), replace = True)
            bootstrap_subjects.append(resample_data)

        bootstrap_data = pd.concat(bootstrap_subjects)
        bootstrap_fit = fit_model(bootstrap_data)
        bootstrap_results.append(bootstrap_fit)
    return bootstrap_results
results = bootstrap(df, 3)
print(results)

assert len(results) == 3



