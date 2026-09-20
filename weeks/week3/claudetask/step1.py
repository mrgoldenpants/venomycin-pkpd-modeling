import numpy as np
from scipy.integrate import solve_ivp


# this is the normal two_compartment_ode
def two_compartment_ode(t, y, CL, Q, V1, V2):
    A1, A2 = y
    dA1_dt = -CL/V1 * A1 - Q * A1 / V1 + Q * A2 / V2
    dA2_dt = Q * A1 / V1 - Q * A2 / V2
    return np.array([dA1_dt, dA2_dt])
#setting the parameters
CL = 1
Q = 2
V1 = 10
V2 = 20

y0 = np.array([100, 0])
t0 = 0
dt = 1
#plug these values into the two_compartment_ode, and store the results into derevative
derevative = two_compartment_ode(t0, y0, CL, Q, V1, V2)
#the euler is taking the initial value and then adding the rate of change
y_euler = y0 + derevative * dt
#so below is the euler method
print("A1 = ", y_euler[0])
print("A2 = ", y_euler[1])
#this is like, the normal diffeq method
solution = solve_ivp(
    two_compartment_ode,
    [0,1 ],
    y0,
    args = [CL, Q, V1, V2],
)

y_solve_ivp = solution.y[:, -1]
print("\nsolve_ivp result after 1 hour:")
print("A1 =", y_solve_ivp[0])
print("A2 =", y_solve_ivp[1])
print("\nComparison:")
#comparing the euler and diffeq A1s
print("Euler A1:", y_euler[0])
print("solve_ivp A1:", y_solve_ivp[0])
#comparing the euler and diffeq A2s
print("Euler A2:", y_euler[1])
print("solve_ivp A2:", y_solve_ivp[1])