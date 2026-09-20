import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

A = np.array([1, 2, 3, 4, 5])
B = np.array([6, 7, 8, 9, 10])

def absolute_error(A, B):
    return np.abs(A - B)

def relative_error(A, B):
    return np.abs(A - B) / np.abs(B)

def rmse(A, B):
    return np.sqrt(np.mean((A - B) ** 2))

print(absolute_error(A, B))
print(relative_error(A, B))
print(rmse(A, B))