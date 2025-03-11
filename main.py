import h5py
import sympy as sp
from check_data import dat_array, print_struc
import numpy as np
import matplotlib.pyplot as plt

# Load data
DeltaAil = dat_array("run8/aircraft/DeltaAil")
IservoAil = dat_array("run8/aircraft/IservoAil")

# Parametric constants
c1 = 2.95  # damper constant TUNING PARAMETER
k1 = 9.9  # spring constant TUNING PARAMETER
kg = 0.45  # gain SET PARAMETER
Ie = 0.4  # moment of inertia TUNING PARAMETER

# Define symbolic matrices
A = sp.Matrix([[-(c1/Ie), -(k1/Ie)], 
               [1, 0]])
B = sp.Matrix([[-kg/Ie], 
               [0]])


# Initial conditions
x = sp.Matrix([[0], [0.013]])  # Initial state (angle and velocity)
xlist = []

# Time integration loop
dt = 0.001  # Time step
for i in range(7001):
    u = sp.Matrix([[IservoAil[i]]])  # Control input
    
    # Compute xdot = A * x + B * u
    xdot = A @ x + B @ u
    #print(xdot)
    x = x + xdot * dt  # Euler integration step

    # Store angle (first element of x)
    xlist.append(x[1])

    #if i % 1000 == 0:
        #print(f"Step {i}: x = {x}")
error = sum((xlist-DeltaAil)**2)
# Plot results
plt.plot(np.linspace(0, 7000, 7001), xlist, color="r", label="Computed")
plt.plot(np.linspace(0, 7000, 7001), DeltaAil, color="b", label="Actual")
plt.legend()
print(error)
plt.show()
