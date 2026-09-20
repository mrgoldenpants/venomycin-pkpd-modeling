import numpy as np
Emax = 5
E0 = 1
C = 10
EC50 = 2.5
AUC = 200
AUC50 = 100


def emax_response(Emax, E0, EC50, C):
    E = E0 + (Emax * C) / (EC50 + C)
    return E

def auc_response(Emax, E0, AUC50, AUC):
    E = E0 + (Emax * AUC) / (AUC50 + AUC)
    return E

concentration_response = emax_response(Emax, E0, EC50, C)
auc_based_response = auc_response(Emax, E0, AUC50, AUC)
