import numpy as np

def calculate_half_life(ke: float) -> float :
    return np.log(2)/ke

ke = np.array([0.05, 0.20, 0.60])

print(calculate_half_life(ke))