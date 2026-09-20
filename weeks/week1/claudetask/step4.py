import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 24, 100)
first = 100*np.exp(-.1*t)
second = 100*np.exp(-.3*t)

plt.figure(figsize = (10,5))

plt.plot(t, first, label = "first", color = "blue", linewidth = 2)
plt.plot(t, second, label = "second", color = "red", linewidth = 2)
plt.xlabel("Time (hours)")
plt.ylabel("Concentration (mg/L)")
plt.title("Concentration vs Time with different ke values")
plt.legend()
plt.show()

#these graphs make sense because they start at 100 mg/L, and exponetially decreases towards zero
#the larger ke produces faster elimination (has a stepper slope)
