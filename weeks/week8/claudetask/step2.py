import numpy as np
from scipy.integrate import solve_ivp
CL = 5
Q = 2
V1 = 10
V2 = 20
Dose = 100
y0 = ([100, 0])
t_eval = np.linspace(0, 500, 1000)


#normal two compartment ode
def two_compartment_ode(t, y, CL, Q, V1, V2):
    A1, A2 = y
    dA1_dt = -CL/V1 * A1 - Q * A1 / V1 + Q * A2 / V2
    dA2_dt = Q * A1 / V1 - Q * A2 / V2
    return np.array([dA1_dt, dA2_dt])

#normal solution

solution = solve_ivp(
        two_compartment_ode,
        [0,500],
        y0,
        args = (CL, Q, V1, V2),
        t_eval = t_eval
)
#results
A1 = solution.y[0]
A2 = solution.y[1]
C1 = A1/V1
total_mass = A1 + A2
numerical_AUC = np.trapezoid(C1, solution.t)
analytical_AUC = Dose/CL
print("Numerical AUC: ", numerical_AUC)
print("Analytical AUC: ", analytical_AUC)
assert np.isclose(numerical_AUC, analytical_AUC, rtol=1e-2)
