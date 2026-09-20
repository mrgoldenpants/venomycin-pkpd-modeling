import numpy as np

def analytical_concentration(t, c0, ke): return c0 * np.exp(-ke * t)
c0 = np.array([0, 2, 4, 6, 8, 10])

assert np.array_equal(analytical_concentration(0, c0, 5), c0)


