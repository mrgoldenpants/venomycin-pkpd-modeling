import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

C0 = 25      # mg/L
ke = 0.12    # 1/hr
t_span = (0, 48)

def analytical_conc(t, C0, ke):
    # TODO: implement C(t) = C0 * e^(-ket)
    return C0*np.exp(-ke*t)


def dCdt(t, C, ke):
    # TODO: implement dC/dt = -keC
    # remember: solve_ivp passes C as an array-like, even for one state
    return -ke*C


def half_life(ke):
    # TODO: implement t1/2 = ln(2)/ke
    return np.log(2)/ke

def run_numerical_solution(C0, ke, t_span, t_eval):
    # TODO: call solve_ivp here with dCdt, t_span, y0=[C0], t_eval, args=(ke,)
    solution = solve_ivp(
        dCdt,
        t_span,
        [C0],
        args = (ke,),
        t_eval = t_eval


    )
    return solution


def compare_solutions(t_eval, analytical, numerical):
    # TODO: compute max absolute error and RMSE between the two arrays
    error = np.abs(numerical.y[0] - analytical)
    max_error = np.max(error)
    rmse = np.sqrt(np.mean((numerical.y[0] - analytical)**2))
    return max_error, rmse



#threetest
def test_analytical_at_t0():
    result = analytical_conc(0, C0, ke)
    assert result == C0  # you already reasoned through this one

def test_half_life():
    t_half = half_life(ke)
    result = analytical_conc(t_half, C0, ke)
    assert abs(result - C0/2) < 1e-6  # what should the concentration equal at t_half, and what tolerance is reasonable?

def test_numerical_matches_analytical():
    t_eval = np.linspace(*t_span, 100)
    numerical = run_numerical_solution(C0, ke, t_span, t_eval)
    analytical = analytical_conc(t_eval, C0, ke)
    max_error = np.max(np.abs(numerical.y[0] - analytical))
    assert max_error < .02  # you already have a real number for this — 0.0099. What threshold makes sense given that?

#--- run it ---
#TODO: generate t_eval, compute analytical curve, compute numerical curve,
#plot both, compute half-life, compute and print the error metrics

t_eval = np.linspace(*t_span, 100)
numerical = run_numerical_solution(C0, ke, t_span, t_eval)
analytical = analytical_conc(t_eval, C0, ke)

plt.figure(figsize = (10,10))
plt.plot(t_eval, analytical, label = "Analytical")
plt.plot(numerical.t, numerical.y[0], label = "Numerical")
plt.xlabel("Time (Hours)")
plt.ylabel("Concentration (mg/L)")
plt.title("Concentration vs Time with Analytical and Numerical")
plt.legend()
plt.show()
t_half = half_life(ke)
print("Half-Life:", t_half, "hours")
max_error, rmse = compare_solutions(numerical.t, analytical, numerical)
print("Maximum absolute error", max_error)
print("RMSE", rmse)

test_analytical_at_t0()
test_half_life()
test_numerical_matches_analytical()
print("All tests passed")
