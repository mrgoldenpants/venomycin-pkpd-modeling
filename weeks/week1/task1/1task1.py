import numpy as np

#create arrays and check their shape
#linspace creates an array of evenly spaced numbers array
time_steps = np.linspace(0, 10, 50)
print(time_steps)

zero_weights = np.zeros(10)
print(zero_weights)

print("Shape:", zero_weights.shape)
print("Shape:", time_steps.shape)
print("Data type:", time_steps.dtype)
print("Data type:", zero_weights.dtype)

