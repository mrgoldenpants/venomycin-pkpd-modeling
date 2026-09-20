import numpy as np
Emax = 5
E0 = 1
EC50 = 2.5
C = EC50

def emax_response(Emax, E0, EC50, C):
    E = E0 + (Emax * C) / (EC50 + C)
    return E

assert emax_response(Emax, E0, EC50, C) == E0 +Emax/2