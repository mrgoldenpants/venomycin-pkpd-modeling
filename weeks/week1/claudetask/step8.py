import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
C0 = 100
ke = 0.1



def dCdt(t, C, ke):
    return -ke*C



t = np.linspace(0, 48, 100)

solution = solve_ivp(
    dCdt,
    [0, 48],
    [C0],
    args = (ke,),
    t_eval = t,
)

#solution.y[0] is the output from solution
def analytical_conc(t, C0, ke):
    return C0* np.exp(-ke*t)

analytical = analytical_conc(t, C0, ke)
print(t, analytical)

#plotting now
plt.figure(figsize=(10,5))
plt.plot(solution.t, solution.y[0], label = "Numerical")
plt.plot(solution.t, analytical, label = "Analytical")
plt.legend()
plt.show()

error = np.abs(solution.y - analytical)
max_error = np.max(error)
print(error, max_error)