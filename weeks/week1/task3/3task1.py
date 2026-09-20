import numpy as np

def calculate_concentration(C0: float, ke: float, t: np.ndarray) -> np.ndarray :
    return C0 * np.exp(-ke * t)

concentration = calculate_concentration(100, 0.15, np.linspace(0, 24, 100))
print(concentration)
concentration = calculate_concentration(100, 0.15, 0)
print(concentration)
concentration = calculate_concentration(100, 0.15, 12)
print(concentration)
concentration = calculate_concentration(100, 0.15, 24)

print(concentration)


