import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

A = np.array([1, 2, 3, 4, 5])
B = np.array([1, 2, 3, 4, 5])

def absolute_error(A, B):
    return np.abs(A - B)

def relative_error(A, B):
    return np.abs(A - B) / np.abs(B)

def rmse(A, B):
    return np.sqrt(np.mean((A - B) ** 2))

assert rmse(A, B) == 0.0
assert np.all(relative_error(A, B) == 0)
assert np.all(absolute_error(A, B) == 0)


