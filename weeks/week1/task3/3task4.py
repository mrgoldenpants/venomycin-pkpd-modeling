import numpy as np

def estimate_ke(t1: float, C1: float, t2: float, C2: float) -> float :
    return ((np.log(C1) - np.log(C2)) / (t2 - t1))

blood_concentration = estimate_ke(2, 80, 8, 20)
print(blood_concentration)

