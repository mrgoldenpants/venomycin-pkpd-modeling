import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from week2.claudetasks.harderversionofstep2 import V1

t = [0, 48]
def two_compartment_ode(t, y, CL, Q, V1, V2):
    A1, A2 = y
    dA1_dt = -CL/V1 * A1 - Q * A1 / V1 + Q * A2 / V2
    dA2_dt = Q * A1 / V1 - Q * A2 / V2
    return np.array([dA1_dt, dA2_dt])

solution = solve_ivp(
    two_compartment_ode,
    t,
    [100, 0],
    args = (5, 2, 10, 20 ),
)

A1 = solution.y[0]
A2 = solution.y[1]
#how do i get the first value of A1
assert A1[0] == 100
assert A2[0] == 0

C1 = A1/V1

plt.figure(figsize=(8, 4))
plt.plot(solution.t, C1, label = "C1 vs time")
plt.xlabel("Time (Hours)")
plt.ylabel("C1 (mg/L)")
plt.legend()
plt.show()
