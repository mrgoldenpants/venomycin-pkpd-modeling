import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
y0 = np.array([100, 0])

def two_compartment_ode(t, y, CL, Q, V1, V2):
    A1, A2 = y
    dA1_dt = -CL/V1 * A1 - Q * A1 / V1 + Q * A2 / V2
    dA2_dt = Q * A1 / V1 - Q * A2 / V2
    return np.array([dA1_dt, dA2_dt])

solution_RK45 = solve_ivp(
        two_compartment_ode,
        [0,48],
        y0,
        args = (5, 2, 10, 20 ),
    method = 'RK45'
)

solution_radau = solve_ivp(
        two_compartment_ode,
        [0,48],
        y0,
        args = (5, 2, 10, 20 ),
    method = 'Radau'
)

plt.plot(solution_RK45.t, solution_RK45.y[0])
plt.plot(solution_radau.t, solution_radau.y[0])
plt.xlabel('Time (Hours)')
plt.ylabel('Amount in A1')
plt.legend(['RK45', 'Radau'])
plt.show()

