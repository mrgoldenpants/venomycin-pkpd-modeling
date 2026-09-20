import numpy as np
Emax = 5
E0 = 1
EC50 = 2.5
C = 5000

def emax_response(Emax, E0, EC50, C):
    E = E0 + (Emax * C) / (EC50 + C)
    return E
E = emax_response(Emax, E0, EC50, C)
assert np.isclose(E, E0 + Emax, rtol = 1e-3)
