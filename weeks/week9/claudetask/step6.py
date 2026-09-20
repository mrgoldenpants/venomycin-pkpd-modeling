import numpy as np

AUC = 500


def calculate_auc_mic(AUC, MIC):
    if MIC <= 0:
        raise ValueError("MIC must be greater than zero")
    return AUC/MIC


try:
    calculate_auc_mic(AUC, -2)
    assert False
except ValueError:
    pass

try:
    calculate_auc_mic(AUC, 0)
    assert False
except ValueError:
    pass