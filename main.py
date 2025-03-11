import numpy as np
import matplotlib.pyplot as plt
from check_data import dat_array

# Load data
DeltaAil = dat_array("run8/aircraft/DeltaAil")
IservoAil = dat_array("run8/aircraft/IservoAil")

# Parametric constants (adjusted for better accuracy)
c1 = 2.65   # Damper constant (TUNING PARAMETER)
k1 = 7.41   # Spring constant (TUNING PARAMETER)
kg = 0.4    # Gain (SET PARAMETER)
Ie = 0.45   # Moment of inertia (TUNING PARAMETER)
c2 = 0.00089

# System matrices
A = np.array([[-(c1/Ie), -(k1/Ie)], [1, 0]])
B = np.array([[-kg/Ie], [0]])
C = np.array([[c2/Ie], [0]])

# Initial state [angle, velocity]
x = np.array([[0], [0.006]])
xlist = []

# Time integration loop
dt = 0.001  # Time step
for i in range(7001):
    u = np.array([[IservoAil[i]]])  # Control input
    v = np.array([[86]])  # External force
    
    # Compute xdot = A * x + B * u + C * v
    xdot = A @ x + B @ u + C @ v
    x = x + xdot * dt  # Euler integration step

    # Store angle (first element of x)
    xlist.append(x[1, 0])

# Convert xlist to NumPy array
xlist = np.array(xlist)

# Compute absolute error
absolute_error = np.abs(xlist - DeltaAil)

# Compute accuracy as percentage
error_norm = np.linalg.norm(absolute_error) / np.linalg.norm(DeltaAil)
accuracy = (1 - error_norm) * 100

# Print accuracy
print(f"Model Accuracy: {accuracy:.2f}%")

# Plot computed vs actual Delta Ail
time_steps = np.linspace(0, 7000, 7001)
plt.figure(figsize=(10, 5))
plt.plot(time_steps, xlist, color="r", label="Computed")
plt.plot(time_steps, DeltaAil, color="b", label="Actual")
plt.xlabel("Time Steps")
plt.ylabel("Delta Ail")
plt.legend()
plt.title("Computed vs. Actual Delta Ail")
plt.show()

# Plot absolute error over time
plt.figure(figsize=(10, 5))
plt.plot(time_steps, absolute_error, color="g", label="Absolute Error")
plt.xlabel("Time Steps")
plt.ylabel("Absolute Error")
plt.legend()
plt.title("Absolute Error Over Time")
plt.show()
