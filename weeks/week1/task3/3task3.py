import numpy as np

def calculate_concentration(C0 , ke, t):
    return C0 * np.exp(-ke * t)

t = np.linspace(0, 10, 5)

normal = calculate_concentration(50, .4, t)
impaired = calculate_concentration(50, .1, t)

print("Normal Renal Function: ", normal)
print("Impaired Renal Function: ", impaired)

