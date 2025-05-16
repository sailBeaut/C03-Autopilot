import numpy as np
import matplotlib.pyplot as plt
from Functions_for_sys2 import model2, accuracy_plot_ail, accuracy_plot_elev
from optimizerfunctions import *

#Resolution
resolution = 2
sensitivity = 0.1

#Initial Parameters Elevator
flatten_elev = True         #Constant for all runs
divfactor_elev = 1          #Constant for all runs
clutch_elev = 0             #Constant for all runs
k1_numvalue_elev = 500000   #Constant for all runs
k2_numvalue_elev_int = 21.2
c1_numvalue_elev = 50       #Constant for all runs
c2_numvalue_elev_int = 3.6
k_g_elev = 0.22             #Constant for all runs
a_velo_elev_int = 2.3e-7
flip_elev = 1               #Constant for all runs
flatten_coeff_elev_int = 0.00001810

#On Or Off
ground = False
aileron = False
elevator = True
array = False
extragraphs = False
showmainplots = False
printeigenvalues = False
print_accuracy = True
tries = 1

for attempt in range(tries):
    epoch = 0
    k2_numvalue = k2_numvalue_elev_int
    c2_numvalue = c2_numvalue_elev_int
    a_velo = a_velo_elev_int
    flatten_coeff = flatten_coeff_elev_int
    increment = True
    decrement = False
    continue_parameter_inc = True
    continue_parameter_dec = False
    continue_parameter = True
    increment_or_decrement_list = [0.1, 0.05, 0.000000005, 0.0000001]
    acc_now = 0
    acc_last = 0
    acc_last_last = 0
    continue_list = [1, 1, 1, 1]  # List to keep track of which parameters are still being tested
    epoch_list = [0,0,0,0]  # List to keep track of epochs per parameter
    chosen_parameter = 'Still generating sufficient Epoch Data'
    # Initialize the parameter values
    k2_update = 0
    c2_update = 0
    a_velo_update = 0
    flatten_coeff_update = 0
    delta_acc1, delta_acc2 = 0, 0
    # Initialize the placeholder for parameter values before increment/decrement
    k2_placeholder = 0
    c2_placeholder = 0
    a_velo_placeholder = 0
    flatten_coeff_placeholder = 0

    while True:
        if sum(continue_list) == 0 and np.all(epoch_list < 5):
            print("All parameters have been tried, exiting the loop.")
            break
        elif sum(continue_list) == 0 and np.all(epoch_list >= 5):
            print("All parameters have been tried but change in parameters still apparent, continue with new random parameter.")
            continue_list = [1, 1, 1, 1]  # Reset the continue list
            epoch_list = [0,0,0,0] # Reset the epoch list
        
        if continue_parameter_dec == True:
            acc_last = 0
            acc_last_last = 0
            k2_update = k2_placeholder
            c2_update = c2_placeholder
            a_velo_update = a_velo_placeholder
            flatten_coeff_update = flatten_coeff_placeholder

        


        #Define Accuracy lists
        accuracy_DOF2_elev = []

        #Define the parameters
        if epoch != 0:
            k2_numvalue = k2_update
            c2_numvalue = c2_update
            a_velo = a_velo_update
            flatten_coeff = flatten_coeff_update

        #Run the model
        for run in (4,5,6,7,12,13):
            acc_run_DOF1, acc_run_DOF2 = model2(run, print_accuracy, array, resolution, flatten_elev, flatten_coeff, clutch_elev, flip_elev, divfactor_elev, k_g_elev, k1_numvalue_elev, k2_numvalue, c1_numvalue_elev, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
            # Calculate accuracy for elevator
            accuracy_DOF2_elev.append(acc_run_DOF2)

        # Calculate average accuracy for DOF 2
        acc_now = calculate_average_accuracy_dof2(accuracy_DOF2_elev)

        #Print the accuracies
        print("---------------------------------------------------------------------------")
        print(f"This is attempt number {attempt+1}")
        print(f"This is epoch number {epoch}")	
        print(f"Average Accuracy for DOF 2: {acc_now:.3f}%")
        print(f"Used Parameters: k2={k2_numvalue}, c2={c2_numvalue}, a_velo={a_velo}, flatten_coeff={flatten_coeff}")
        print(f"This is the accuracy of the last epoch: {acc_last:.3f}%")
        print(f"This is the accuracy of the epoch before that: {acc_last_last:.3f}%")
        print(f"Change in accuracy: 0 and -1 {delta_acc1:.3f}% & -1 and -2 {delta_acc2:.3f}%")
        print(f"Parameter chosen: {chosen_parameter}")
        print(f"Increment & decrement: {increment} & {decrement}")
        print(f"Increment or decrement list: {increment_or_decrement_list}")
        print(f"Continue Increment & decrement: {continue_parameter_inc} & {continue_parameter_dec}")
        print(f"Continue parameter: {continue_parameter}")
        print(f"Continue list: {continue_list}")
        print(f"Epoch list: {epoch_list}")

        print("---------------------------------------------------------------------------")
        
        #Add Epoch counter
        epoch += 1

        
        #Change the parameters
        if continue_parameter == False or epoch == 1:
            continue_parameter = True
            chosen_parameter = choose_random_param(continue_list)
            k2_update, c2_update, a_velo_update, flatten_coeff_update = increment_or_decrement_parameter(chosen_parameter, increment, decrement, k2_numvalue, c2_numvalue, a_velo, flatten_coeff,  increment_or_decrement_list)
        elif epoch == 2:
            k2_update, c2_update, a_velo_update, flatten_coeff_update = increment_or_decrement_parameter(chosen_parameter, increment, decrement, k2_update, c2_update, a_velo_update, flatten_coeff_update, increment_or_decrement_list)
        
        #Increment or decrement the parameters
        if epoch > 2 and continue_parameter == True:
            k2_update, c2_update, a_velo_update, flatten_coeff_update =  increment_or_decrement_parameter(chosen_parameter, continue_parameter_inc, continue_parameter_dec, k2_update, c2_update, a_velo_update, flatten_coeff_update, increment_or_decrement_list)
   
    
      
        #Compare the accuracies
        if increment == True:
            if epoch > 0:            #store the current parameter values before incrementing/decrementing
                k2_placeholder = k2_update
                c2_placeholder = c2_update
                a_velo_placeholder = a_velo_update
                flatten_coeff_placeholder = flatten_coeff_update

        if epoch > 2:
            delta_acc1, delta_acc2 = calculate_accuracy_change_3step(acc_now, acc_last, acc_last_last)
            # Check if the changes in accuracy are within the sensitivity range
            # If both changes are within the sensitivity range, continue with the current parameter
            increment, decrement = continue_parameter_inc, continue_parameter_dec
            continue_parameter_inc, continue_parameter_dec, continue_parameter = compare_accuracies_and_choose_to_continue(delta_acc1, delta_acc2, sensitivity, increment, decrement, continue_parameter_inc, continue_parameter_dec, continue_parameter)
            if continue_parameter == False:
                continue_list[chosen_parameter] = 0
            # Removed incomplete if statement to fix syntax error

        #Save the accuracies
        acc_last_last = acc_last
        acc_last = acc_now
        

        #Count Epochs per parameter
        epoch_list = count_epochs_per_parameter(chosen_parameter, continue_list, epoch_list)
       