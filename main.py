import numpy as np
import matplotlib.pyplot as plt
from check_data import dat_array as load_data
import os

# Load the data for a specific run
Accuracy = []
for i in [1]:
    nr_of_run = i
    run_nr = nr_of_run
    DeltaAil = load_data("run" + str(run_nr) + "/aircraft/DeltaAil")
    IservoAil = load_data("run" + str(run_nr) + "/aircraft/IservoAil")
    IservoAil-=IservoAil[0]

    # Tuning Parameters 
    c1 = 0.94  # Damper constant 
    k1 = 3.75   # Spring constant 

    # Set Parameters
    kg = 0.22 # Gain 
    Ie = 0.001 # Moment of inertia 

    # System matrices
    A = np.array([[-(c1/Ie), -(k1/Ie)], [1, 0]]) 
    B = np.array([[-kg/Ie], [0]]) 

    x = np.array([[0], [0]]) # Initial state [angle, velocity]
    xlist = [] # List to store state vectors

    # Time integration loop
    dt = 0.001  # Time step (1ms)
    for i in range(len(DeltaAil)):
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
    '''
    time_steps = np.linspace(0, len(DeltaAil)-1, len(DeltaAil))
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, xlist, color="r", label="Computed")
    plt.plot(time_steps, DeltaAil, color="b", label="Actual")
    plt.xlabel("Time Steps")
    plt.ylabel("Delta Ail")
    plt.legend()
    plt.title("Computed vs. Actual Delta Ail")
    plt.show()
    '''
    
    t_values = np.linspace(0, len(DeltaAil)-1, len(DeltaAil))
    plt.plot(t_values, xlist, label=r'$\theta$' + ": predicted by SDOF Model", color="blue")
    plt.plot(t_values, DeltaAil, label=r'$\theta$' + ": actual data", color="orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement of " + r'$\theta$')
    plt.title(r'$\theta$' + " vs Time - Run 3")# (Worst Accuracy)")
    plt.text(x=7000, y=-0.033, s="Model accuracy of " + r'$\theta$' + f": {accuracy:.2f}%", fontsize=10, color="black")
    plt.legend()
    plt.grid()
    plt.savefig("SDOF_ail_3.png", dpi=300, bbox_inches='tight')
    print("Saving to:", os.path.abspath("SDOF_ail_1.png"))
    plt.show()
    

lennart = sum(Accuracy)/len(Accuracy)
print(lennart)

# Plot computed vs actual Delta Ail over time


