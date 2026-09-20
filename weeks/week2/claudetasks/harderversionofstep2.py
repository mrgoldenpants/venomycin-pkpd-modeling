import numpy as np

#a1/dt = -keA1 - Q*A1/V1 + Q*A2/V2
#a2/dt = Q*A1/V1 - Q*A2/V2

Q = 0


A1 = 100
A2 = 200
V1 = 150
V2 = 250
ke = .1

dA1_dt = -ke*A1 - Q*A1/V1 + Q*A2/V2
dA2_dt = Q*A1/V1 - Q*A2/V2

assert dA1_dt + dA2_dt == -ke*A1
