import numpy as np
Q = 0
CL = 0
def two_compartment_ode(t, y, CL, Q, V1, V2):
    A1, A2 = y
    dA1_dt = -CL/V1 * A1 - Q * A1 / V1 + Q * A2 / V2
    dA2_dt = Q * A1 / V1 - Q * A2 / V2
    return np.array([dA1_dt, dA2_dt])


result = two_compartment_ode(10, [100,200], CL, Q, 5, 6)
assert result[0] == 0
assert result[1] == 0