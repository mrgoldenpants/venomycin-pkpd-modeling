import numpy as np
#convert celsius to fahrenheit

celsius = np.array([0, 15, 22, 37, 100])
print(celsius)
fahrenheit = np.array((celsius * 9 / 5) + 32)
print(fahrenheit)