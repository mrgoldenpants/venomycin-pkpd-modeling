import numpy as np
from scipy.integrate import solve_ivp



def two_compartment_ode(t, y, CL, Q, V1, V2):
    A1, A2 = y
    dA1_dt = -CL/V1 * A1 - Q * A1 / V1 + Q * A2 / V2
    dA2_dt = Q * A1 / V1 - Q * A2 / V2
    return np.array([dA1_dt, dA2_dt])

CL = 1
Q = 2
V1 = 10
V2 = 20

y0 = np.array([100, 0])
t0 = 0
dt = .001

derevative = two_compartment_ode(t0, y0, CL, Q, V1, V2)

y_euler = y0 + derevative * dt

print("A1(using euler method) = ", y_euler[0])
print("A2(using euler method) = ", y_euler[1])
#below we are creating the diffeq/non euler method
solution = solve_ivp(
    two_compartment_ode,
    [0,dt ],
    y0,
    args = [CL, Q, V1, V2],
    max_step =dt
)

y_solve_ivp = solution.y[:, -1]
#np.allclose is saying the values in the array are all close
assert np.allclose(y_euler, y_solve_ivp, atol=1e-5)