import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 24, 300)
C = 100*np.exp(-0.2*t) + 100*np.exp(-0.2*(t-8))*(t>=8) + 100*np.exp(-0.2*(t-16))*(t>=16)

plt.grid(True)
plt.axvline(x=8, color = 'red', linestyle = '--', linewidth = 3)
plt.axvline(x=16, color = 'red', linestyle = '--', linewidth = 3)
plt.plot(t, C)
plt.xlabel("Time (hours)")
plt.ylabel("Concentration (mg/L)")
plt.show()