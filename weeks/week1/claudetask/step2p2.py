import numpy as np

def f(x):
   return x**2 + 4*x +1

x_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_values = f(x_values)

print(x_values)
print(y_values)

assert f(2) == 13


