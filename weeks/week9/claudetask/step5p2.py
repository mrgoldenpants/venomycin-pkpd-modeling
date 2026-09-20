import numpy as np

AUC = 500
MIC = 1

def calculate_auc_mic(AUC, MIC):
    return AUC/MIC

x = calculate_auc_mic(AUC, MIC)
y = calculate_auc_mic(AUC, 2*MIC)
assert 1/2*x == y


