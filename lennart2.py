import numpy as np
import matplotlib.pyplot as plt
from check_new_data import dat_array as load_data

# Load the data for a specific run
Accuracy = []
if True:
    nr_of_run = 1
    run_nr = nr_of_run
    DeltaAil = load_data("run1/DeltaAil")
    IservoAil = load_data("run1/IservoAil")

    # Tuning Parameters 
    c1 = 1.29  # Damper constant 
    k1 = 4.23   # Spring constant 

    # Set Parameters
    kg = 0.22 # Gain 
    Ie = 0.0451 # Moment of inertia

    # System matrices
    A = np.array([[-(c1/Ie), -(k1/Ie)], [1, 0]]) 
    B = np.array([[-kg/Ie], [0]]) 

    x = np.array([[0], [0]]) # Initial state [angle, velocity]
    xlist = [] # List to store state vectors

    # Time integration loop
    dt = 0.001  # Time step (1ms)
    for i in range(7001):
        u = np.array([[IservoAil[i]]]) # Control current input    

        xdot = A @ x + B @ u # Compute the derivative from system matrices and inputs
        x = x + xdot * dt  # Euler integration step

        xlist.append(x[1, 0]) # Store control surface deflection

    # Convert xlist to NumPy array
    xlist = np.array(xlist)

    # Compute absolute error
    absolute_error = np.abs(xlist - DeltaAil)

    # Compute relative error and accuracy as percentages
    error_norm = np.linalg.norm(absolute_error) / np.linalg.norm(DeltaAil)
    accuracy = (1 - error_norm) * 100
    Accuracy.append(accuracy)

    # Print accuracy
    print(f"Model Accuracy: {accuracy:.2f}%")

lennart = sum(Accuracy)/len(Accuracy)
print(lennart)

# Plot computed vs actual Delta Ail over time
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