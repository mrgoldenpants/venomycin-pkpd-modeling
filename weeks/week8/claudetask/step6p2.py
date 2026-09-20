import numpy as np
from scipy.integrate import solve_ivp
t = [0, 2, 4, 6, 8]
C = [10, 7, 12, 8, 4]
Dose = 100
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
C1 = A1/V1

def calculate_auc(t, c):
    total_auc = 0.0
    for i in range(len(t) - 1):
        time_width = t[i + 1] - t[i]
        average = (c[i] + c[i + 1]) / 2
        area = time_width * average
        total_auc += area

    return total_auc

def calculate_cmax(c):
    cmax = 0
    for i in range(len(c)):
        if c[i] > cmax:
            cmax = c[i]
    return cmax

def calculate_tmax(t, c):
    return t[np.argmax(c)]


def calculate_half_life(CL, V1):
    return (V1*np.log(2))/CL

assert calculate_cmax(C) == 12
assert calculate_tmax(t,C) == 4
assert np.isclose(calculate_half_life(5, 10), 1.38629436112)
assert calculate_auc(t, C) == 68