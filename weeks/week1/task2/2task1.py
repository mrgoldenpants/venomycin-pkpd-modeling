import numpy as np

def calculate_position(d0, v0, a, t):
    return d0 + v0*t + 1/2*a*(t**2)

position = calculate_position(10.0, 3.0, 2.0, 5.0)


t_first = 5.0
position = calculate_position(10, 3.0, 2.0, t_first)

t_second = np.linspace(0, 10, 100)
position = calculate_position(10, 3.0, 2.0, t_second)
print(position)