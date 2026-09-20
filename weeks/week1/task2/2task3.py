import numpy as np
#np.maxmimum compares each element of the array
#so np.maxmimum(base number your comparing to, each number in array)
#so it goes np.maxmimum(0, -5) and outputs the max between 0 and -5 which is 0.
def relu(x):
    return np.maximum(0, x)

x = np.array([-5, -2, 0, 3, 8])

print(relu(x))