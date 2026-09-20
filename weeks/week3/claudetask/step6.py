import numpy as np
from scipy.integrate import solve_ivp
CL = 5
V1 = 10
Dose = 100
t = np.linspace(0, 48, 100)
def analytical_one_compartment(t, CL, Dose, V1):
    return Dose * np.exp((-CL/V1) * t)

A1_analytical = analytical_one_compartment(t, CL, Dose, V1)
print(A1_analytical)
