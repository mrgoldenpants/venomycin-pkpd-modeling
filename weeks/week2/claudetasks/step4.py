import numpy as np

Q = 0
C = 0
L = 0
dA1_dt = -(C*L/V1)*A1 - Q*A1/V1 + Q*A2/V2
dA2_dt = Q*A1/V1 - Q*A2/V2
#so the derivative vector becomes [0, 0]
