import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
initial_guess = [4, 4, 4, 4]
df = pd.read_csv("/Users/goldenpants/PycharmProjects/PythonProject/pk_dataset.csv")
lower_bounds = [0.05, 0.01, 3, 3]
upper_bounds = [15, 15, 20, 30]


def two_compartment_ode(t, y, CL, Q, V1, V2):
    A1, A2 = y
    dA1_dt = -CL / V1 * A1 - Q * A1 / V1 + Q * A2 / V2
    dA2_dt = Q * A1 / V1 - Q * A2 / V2
    return np.array([dA1_dt, dA2_dt])
def residuals(theta, observed_time, observed_concentration):
    CL, Q, V1, V2 = theta
    y0 = np.array([100, 0])
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
results = bootstrap(df, 100)
assert len(results) == 100
plot_time = np.linspace(0, 48, 100)
first_draw = results[0]
print(first_draw)
first_CL = first_draw["CL"].iloc[0]
first_Q = first_draw["Q"].iloc[0]
first_V1 = first_draw["V1"].iloc[0]
first_V2 = first_draw["V2"].iloc[0]
first_prediction = solve_ivp(
    two_compartment_ode,
    [0, 48],
    [100, 0],
    args = (first_CL, first_Q, first_V1, first_V2),
    t_eval = plot_time
)
first_A1 = first_prediction.y[0]
first_concentration = first_A1 / first_V1

prediction_curves = []
for draw in results:
    CL = draw["CL"].iloc[0]
    Q = draw["Q"].iloc[0]
    V1 = draw["V1"].iloc[0]
    V2 = draw["V2"].iloc[0]
    prediction = solve_ivp(
        two_compartment_ode,
        [0, 48],
        [100, 0],
        args=(CL, Q, V1, V2),
        t_eval = plot_time
    )
    A1 = prediction.y[0]
    concentration = A1 / V1
    prediction_curves.append(concentration)
prediction_curves = np.array(prediction_curves)
lower_band = np.percentile(prediction_curves, 2.5, axis = 0)
upper_band = np.percentile(prediction_curves, 97.5, axis = 0)
observed_time = df["Time"].values
observed_concentration = df["C"].values
plt.fill_between(plot_time, lower_band, upper_band, alpha = 0.3, label = "95% bootstrap band")
plt.scatter(observed_time, observed_concentration, label = "observed")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Bootstrap band")
plt.legend()
plt.show()