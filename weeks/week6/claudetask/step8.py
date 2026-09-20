import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
df = pd.read_csv("pk_dataset_3.csv")

initial_guess = [4, 4, 4, 4]
lower_bounds = [0.05, 0.01, 3, 3]
upper_bounds = [15, 15, 20, 30]

def fit_robust(data, initial_guess, lower_bounds, upper_bounds):
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
            result = least_squares(residuals, initial_guess, args=(observed_time, observed_concentration),
                                   bounds=(lower_bounds, upper_bounds))
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

    return fit_model(data)

def test_reproducibility_fit():
    np.random.seed(42)
    result_1 = fit_robust(df, initial_guess, lower_bounds, upper_bounds)
    np.random.seed(42)
    result_2 = fit_robust(df, initial_guess, lower_bounds, upper_bounds)
    pd.testing.assert_frame_equal(result_1, result_2)

