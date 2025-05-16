import numpy as np
import matplotlib.pyplot as plt
from check_data import dat_array as load_data
import os

# Load the data for a specific run
Accuracy = []
for i in [4,5,6,7,12,13]:
    nr_of_run = i
    run_nr = nr_of_run
    DeltaAil = load_data("run" + str(run_nr) + "/aircraft/DeltElev")
    IservoAil = load_data("run" + str(run_nr) + "/aircraft/IservoElev")
    IservoAil-=IservoAil[0]

    # Tuning Parameters 
    c1 = 1.1  # Damper constant 3.5
    k1 = 9.4   # Spring constant 52

    # Set Parameters
    kg = -0.22 # Gain 
    Ie = 0.03 # Moment of inertia  0.045

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
    print(accuracy)

    # Print accuracy
    '''
    print(f"Model Accuracy: {accuracy:.2f}%")
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
    '''
    t_values = np.linspace(0, len(DeltaAil)-1, len(DeltaAil))/1000
    plt.plot(t_values, xlist, label=r'$\theta$' + ": predicted by SDOF Model", color="blue")
    plt.plot(t_values, DeltaAil, label=r'$\theta$' + ": actual data", color="orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement of " + r'$\theta$')
    plt.title(r'$\theta$' + " vs Time - Run 13")
    #plt.text(x=-500/1000, y=-0.008, s="Model accuracy of " + r'$\theta$' + f": {accuracy:.2f}%", fontsize=10, color="black")
    #plt.text(x=7000/1000, y=0.011, s="Model accuracy of " + r'$\theta$' + f": {accuracy:.2f}%", fontsize=10, color="black")
    plt.text(x=-200/1000, y=-0.011, s="Model accuracy of " + r'$\theta$' + f": {accuracy:.2f}%", fontsize=10, color="black")    
    plt.legend()
    plt.grid()
    plt.savefig("SDOF_ail_13.png", dpi=300, bbox_inches='tight')
    print("Saving to:", os.path.abspath("SDOF_ail_13.png"))
    plt.show()
    '''
    
    

lennart = sum(Accuracy)/len(Accuracy)
print(lennart)

# # Plot computed vs actual Delta Ail over time




# # Plot absolute error over time
# plt.figure(figsize=(10, 5))
# plt.plot(time_steps, absolute_error, color="g", label="Absolute Error")
# plt.xlabel("Time Steps")
# plt.ylabel("Absolute Error")
# plt.legend()
# plt.title("Absolute Error Over Time")
# plt.show()