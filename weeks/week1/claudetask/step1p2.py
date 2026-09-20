import numpy as np
def test_linspace():
    n = 100
    t = np.linspace(0, 48, n)
    assert len(t) == n
    assert t[0] == 0
    assert t[-1] == 48
