import numpy as np
def one_compartment_ode(t, y, CL, V1):
    A1 = y[0]
    dA1_dt = -CL / V1 * A1
    return np.array([dA1_dt])
