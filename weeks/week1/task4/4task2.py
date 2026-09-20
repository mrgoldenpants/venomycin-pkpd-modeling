import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 24, 100)
normal = 100 * np.exp(-.3 * t)
impaired = 100 * np.exp(-.08 * t)

plt.figure(figsize=(10, 8))
plt.plot(t, normal, label="Normal Patient", color="blue", linestyle="-")
plt.plot(t, impaired, label="Impaired Patient", color="red", linestyle="dashed")
plt.xlabel("Time in hours")
plt.ylabel("Concentration in mg/L")
plt.title("Normal vs Impaired Drug Clearance")
plt.legend()
plt.show()



