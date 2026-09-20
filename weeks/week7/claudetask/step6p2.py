import numpy as np
n = 10
k = 2
SSE = 5
expected_AIC = -2.9314718056
expected_BIC = -2.3263016196
predicted_AIC = n*np.log(SSE/n) + 2*k
predicted_BIC = n*np.log(SSE/n) + k*np.log(n)
print(expected_AIC, predicted_AIC)
assert np.isclose(expected_AIC, predicted_AIC)
assert np.isclose(expected_BIC, predicted_BIC)