import h5py
import sympy as sp
from check_data import dat_array, print_struc
import numpy as np
import matplotlib.pyplot as plt

DeltaAil = dat_array("run1/aircraft/DeltaAil")
IservoAil = dat_array("run1/aircraft/IservoAil")
t_values = np.linspace(0,7000,7001)
kg = 1

Torque = IservoAil*kg
print(IservoAil)
print(len(IservoAil))
print(t_values)

plt.plot(t_values, Torque, label="x2 (DOF 2)")
plt.plot(t_values, IservoAil, label="x2 (DOF 2)")
plt.xlabel("Time (s)")
plt.ylabel("Displacement")
plt.title("MDOF System Response (RK4)")
plt.legend()
plt.grid()
plt.show()