import numpy as np
from scipy.integrate import solve_ivp


def dCdt(t, C, ke):
    return -ke*C

C0 = 100
ke = 0.1

t = np.linspace(0, 48, 100)

solution = solve_ivp(
    dCdt,
    [0, 48],
    [C0],
    args = (ke,),
    t_eval = t,
)
print(solution.t) #my time points
print(solution.y) #my concentration at those time points


