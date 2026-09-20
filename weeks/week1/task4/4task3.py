import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 24, 100)
normal = 100 * np.exp(-.3 * t)

plt.figure(figsize=(10, 8))
plt.plot(t, normal, label="Normal Patient", color="blue", linewidth=2)
plt.axhline(y = 20, color = 'r', linestyle = '--', label = 'Minimum Effective Concentration')
plt.axhline(y = 80, color = 'g', linestyle = '--', label = 'Toxicity Threshold')
plt.xlabel("Time in Hours")
plt.ylabel("Concentration in mg/L")
plt.title("Safeness of the Drug")
plt.legend()
plt.show()