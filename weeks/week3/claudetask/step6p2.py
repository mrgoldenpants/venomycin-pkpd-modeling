import numpy as np
from scipy.integrate import solve_ivp

#one compartment
CL = 5
V1 = 10
Dose = 100
t = np.linspace(0, 48, 100)
def analytical_one_compartment(t, CL, Dose, V1):
    return Dose * np.exp((-CL/V1) * t)

A1_analytical = analytical_one_compartment(t, CL, Dose, V1)

#two compartment
Q = 0
y0 = np.array([100, 0])
def two_compartment_ode(t, y, CL, Q, V1, V2):
    A1, A2 = y
    dA1_dt = -CL/V1 * A1 - Q * A1 / V1 + Q * A2 / V2
    dA2_dt = Q * A1 / V1 - Q * A2 / V2
    return np.array([dA1_dt, dA2_dt])

solution = solve_ivp(
        two_compartment_ode,
        [0,48],
        y0,
        args = (CL, Q, 10, 20 ),
        t_eval = t,
        rtol = 1e-9,
    atol = 1e-9,
)

A1 = solution.y[0]
A2 = solution.y[1]
total = A1 + A2

assert np.allclose(A1, A1_analytical, atol = 1e-4)

