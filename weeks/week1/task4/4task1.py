import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 24, 100)
C = 100 * np.exp(-.15 * t)

plt.figure(figsize=(8,5))
plt.plot(t, C, label="100 mg dose", color="blue", linewidth=2)

plt.xlabel("Time in hours")
plt.ylabel("Drug Concentration in mg/L")
plt.title("Drug Concentration Over Time")

plt.show()