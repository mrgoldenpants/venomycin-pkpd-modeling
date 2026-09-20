import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
upper_bounds = [12, 24, 48]
auc_bounds = []
start_time = 12
CL_values = [1, 2, 3, 4, 5]
AUC_values = []

Dose = 100
dosing_interval = 24
simulation_end = 240
all_times = []
all_concentrations = []
#parameters
y0 = np.array([Dose, 0])
V1 = 10

#normal two compartment ode
def two_compartment_ode(t, y, CL, Q, V1, V2):
    A1, A2 = y
    dA1_dt = -CL/V1 * A1 - Q * A1 / V1 + Q * A2 / V2
    dA2_dt = Q * A1 / V1 - Q * A2 / V2
    return np.array([dA1_dt, dA2_dt])

#normal solution
solution = solve_ivp(
        two_compartment_ode,
        [0,48],
        y0,
        args = (5, 2, V1, 20 ),
)
#results
A1 = solution.y[0]
A2 = solution.y[1]
C1 = A1/V1
total_mass = A1 + A2

def trapezoid_auc(t, c):
    total_auc = 0.0
    for i in range(len(t) - 1):
        time_width = t[i + 1] - t[i]
        average = (c[i] + c[i + 1]) / 2
        area = time_width * average
        total_auc += area

    return total_auc
AUC = trapezoid_auc(solution.t, C1)
def auc_24_hour_window(t, c, start_time):
    end_time = start_time + 24
    mask = (t >= start_time) & (t <= end_time)
    time_subset = t[mask]
    concentration_subset = c[mask]
    return trapezoid_auc(time_subset, concentration_subset)

for CL in CL_values:
    y0 = np.array([Dose, 0])

    solution = solve_ivp(
        two_compartment_ode,
        [0, 24],
        y0,
        args=(CL, 2, V1, 20),
        t_eval = np.linspace(0, 24, 100)
    )
    A1 = solution.y[0]
    C1 = A1/V1
    auc = trapezoid_auc(solution.t, C1)
    AUC_values.append(auc)

plt.plot(CL_values, AUC_values)
plt.xlabel("Clearance")
plt.ylabel("AUC")
plt.title("AUC vs Clearance")
plt.show()

