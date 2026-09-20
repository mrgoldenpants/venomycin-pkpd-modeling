import numpy as np
from scipy.integrate import solve_ivp
CL = .5
V1 = 10
A0 = 100
t = np.linspace(0, 48, 100)
def one_compartment_ode(t, y, CL, V1):
    A1 = y[0]
    dA1_dt = -CL / V1 * A1
    return np.array([dA1_dt])

solution = solve_ivp(
    one_compartment_ode,
    t_span=[0, 48],
    y0 = [A0],
    args = (CL, V1),
    t_eval = t,
    rtol = 1e-8,
    atol = 1e-8
)
analytical = A0 * np.exp((-CL/V1)*t)
assert np.allclose(analytical, solution.y[0], rtol=1e-3)

