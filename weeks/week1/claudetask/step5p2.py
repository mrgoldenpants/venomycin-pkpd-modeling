import numpy as np


C0 = 5
ke = .1
#t = half life t
#analytical concentration formula
def analytical_concentration(t, C0, ke):
    return C0 * np.exp(-ke * t)

half_life = np.log(2)/ke

assert np.isclose(analytical_concentration(half_life, C0, ke), C0/2, atol=1e-6)


