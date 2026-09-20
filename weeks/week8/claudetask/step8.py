import numpy as np
from scipy.integrate import solve_ivp
t = np.array([0, 2, 4])
c = np.array([10, 6, 4])

def trapezoid_auc(t, c):
    total_auc = 0.0
    for i in range(len(t) - 1):
        time_width = t[i + 1] - t[i]
        average = (c[i] + c[i + 1]) / 2
        area = time_width * average
        total_auc += area

    return total_auc

my_auc = trapezoid_auc(t, c)
numpy_auc = np.trapezoid(c, t)