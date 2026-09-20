import numpy as np

def logistic_growth(t, K = 1000, P0 = 10, r = 0.2):
    return K/(1 + ((K-P0)/P0)*(np.exp(-r*t)))

t = np.array([0, 5, 10, 20, 50])
growth = logistic_growth(t)
print(growth)