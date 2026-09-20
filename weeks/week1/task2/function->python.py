import numpy as np
#turning mathematical functions into python

def f(x):
    return x**3 + 4*x + 10

print(f(2))

x_value = np.linspace(-5, 5, 5)
y_value = f(x_value)

print("X values: ", x_value)
print("Y value: ", y_value)
