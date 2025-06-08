import numpy as np
import matplotlib.pyplot as plt
from check_data import dat_array as load_data
import os
# import functions

plot_all = True # creates plots of displacement against time
print_accuracies = True # prints accuracies for all runs, and average


Accuracy = [] # initalizes array for accuracies

for i in [4,5,6,7,12,13]: # loop all aileron runs
    nr_of_run = i
    run_nr = nr_of_run
    DeltaAil = load_data("run" + str(run_nr) + "/aircraft/DeltElev") # load elevator deflection and current data
    IservoAil = load_data("run" + str(run_nr) + "/aircraft/IservoElev")
    IservoAil-=IservoAil[0] # normalize current

    # Tuning Parameters 
    c1 = 1.1  # Damper constant 
    k1 = 9.4   # Spring constant
    Ie = 0.03 # Moment of inertia 

    # Set Parameters
    kg = -0.22 # Gain 


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
    # Convert xlist to NumPy array
    xlist = np.array(xlist)


    absolute_error = np.abs(xlist - DeltaAil) # Compute absolute error
    error_norm = np.linalg.norm(absolute_error) / np.linalg.norm(DeltaAil) # Compute relative error and accuracy as percentages
    accuracy = (1 - error_norm) * 100
    Accuracy.append(accuracy)
    if print_accuracies: # print accuracy if enabled
        print(f"Run {nr_of_run} accuracy: {accuracy:.2f}%") # Print accuracy



    if plot_all: # plot all runs if on 
        t_values = np.linspace(0, len(DeltaAil)-1, len(DeltaAil))/1000 # create timesteps
        plt.plot(t_values, xlist, label=r'$\theta$' + ": predicted by SDOF Model", color="blue") # plot computed displacement 
        plt.plot(t_values, DeltaAil, label=r'$\theta$' + ": actual data", color="orange") # plot actual displacement 
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement of " + r'$\theta$')
        plt.title(r'$\theta$' + " vs Time for Elevator - Run "+str(nr_of_run))
        #plt.text(x=7000/1000, y=-0.033, s="Model accuracy of " + r'$\theta$' + f": {accuracy:.2f}%", fontsize=10, color="black") 
        plt.text(x=-250/1000, y=0.035, s="Model accuracy of " + r'$\theta$' + f": {accuracy:.2f}%", fontsize=10, color="black") # print accuracy in plot, can change position by uncommenting other line
        plt.legend()
        plt.grid()
        plt.show() 

    
if print_accuracies: # print total accuracy if turned on 
    averageacc = sum(Accuracy)/len(Accuracy)
    print(f"Average accuracy: {averageacc:.2f}%")




