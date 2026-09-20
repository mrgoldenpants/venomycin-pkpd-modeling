import numpy as np
from scipy.integrate import solve_ivp
#solutiony is the concentration
#solution.y[0][0] is the very initial condition

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

assert solution.y[0][0] == C0
