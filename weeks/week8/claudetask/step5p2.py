import numpy as np
from scipy.integrate import solve_ivp
upper_bounds = [12, 24, 48]
auc_bounds = []
start_time = 12

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
for bound in upper_bounds:
    mask = solution.t <= bound
    time_subset = solution.t[mask]
    concentration_subset = C1[mask]
    auc = trapezoid_auc(time_subset, concentration_subset)
    auc_bounds.append(auc)

for start  in range(0, simulation_end, dosing_interval):
    end = start + dosing_interval
    t_eval = np.linspace(start, end, 100)
    solution = solve_ivp(
        two_compartment_ode,
        [start, end],
        y0,
        args = (5, 2, V1, 20 ),
        t_eval = t_eval,
    )
    A1 = solution.y[0]
    C1 = A1/V1
    all_times.extend(solution.t)
    all_concentrations.extend(C1)
    final_A1 = solution.y[0, -1]
    final_A2 = solution.y[1, -1]
    y0 = np.array([final_A1 + Dose, final_A2])

all_times = np.array(all_times)
all_concentrations = np.array(all_concentrations)
AUC_192_216 = auc_24_hour_window(all_times, all_concentrations, 192)
AUC_216_240 = auc_24_hour_window(all_times, all_concentrations, 216)
print(AUC_192_216)
print(AUC_216_240)

assert np.isclose(AUC_192_216, AUC_216_240, rtol = 1e-2)
