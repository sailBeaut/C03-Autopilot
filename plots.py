import matplotlib.pyplot as plt
import numpy as np
import os




plt.plot(t_values, flip * Y_sol[:, 1], label=r'$\theta_{2}$' + ": predicted by MDOF Model", color="blue")
plt.plot(t_values, Delta, label=r'$\theta_{2}$' + ": actual data", color="orange")
plt.xlabel("Time (s)")
plt.ylabel("Displacement of " + r'$\theta_{2}$')
plt.title(r'$\theta_{2}$' + " vs Time - Run 3 (Worst Accuracy)")
plt.text(x=0.5, y=0.05, s="Model accuracy of " + r'$\theta_{2}$' + f": {accuracy2:.2f}%", fontsize=10, color="black")
plt.legend()
plt.grid()
plt.savefig("my_plot.png", dpi=300, bbox_inches='tight')
print("Saving to:", os.path.abspath("my_plot.png"))
plt.show()