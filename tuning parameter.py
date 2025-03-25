import h5py
import sympy as sp
from check_data import dat_array, print_struc
import numpy as np
import matplotlib.pyplot as plt
from Functions_for_sys2 import model2_aileron, accuracy_plot

# Parameters
k1_numvalue = 400000
k2_numvalue = 12 
c1_numvalue = 350  # c1 is kept constant
c2_numvalue = 4.5
k_g = 0.22
a_velo = 0.0000004

# On Or Off
extragraphs = False
showmainplots = False
printeigenvalues = False

# Function to calculate average accuracy for DOF 2
def calculate_average_accuracy_dof2(accuracy_DOF2):
    return np.mean(accuracy_DOF2)

# Optimization loop
best_accuracy = 0
best_params = (k1_numvalue, k2_numvalue, c2_numvalue)


while best_accuracy < 85:
    # Run the model with current parameters
    run1_acc1, run1_acc2 = model2_aileron(1, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
    run3_acc1, run3_acc2 = model2_aileron(3, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
    run8_acc1, run8_acc2 = model2_aileron(8, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
    run9_acc1, run9_acc2 = model2_aileron(9, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)    
    run10_acc1, run10_acc2 = model2_aileron(10, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
    run11_acc1, run11_acc2 = model2_aileron(11, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)

    # Accuracy
    accuracy_DOF1 = [run1_acc1, run3_acc1, run8_acc1, run9_acc1, run10_acc1, run11_acc1]
    accuracy_DOF2 = [run1_acc2, run3_acc2, run8_acc2, run9_acc2, run10_acc2, run11_acc2]

    # Calculate average accuracy for DOF 2
    average_accuracy_dof2 = calculate_average_accuracy_dof2(accuracy_DOF2)

    # Check if the current parameters give better accuracy for DOF 2
    if average_accuracy_dof2 > best_accuracy:
        best_accuracy = average_accuracy_dof2
        best_params = (k1_numvalue, k2_numvalue, c2_numvalue)

    # Update parameters for next iteration (example: increment k1, k2, c2)
    #k1_numvalue += 10000
    #k2_numvalue += 1
    c2_numvalue += 0.15

# Use the best parameters found
k1_numvalue, k2_numvalue, c2_numvalue = best_params

# Final run with best parameters
run1_acc1, run1_acc2 = model2_aileron(1, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
run3_acc1, run3_acc2 = model2_aileron(3, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
run8_acc1, run8_acc2 = model2_aileron(8, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
run9_acc1, run9_acc2 = model2_aileron(9, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)    
run10_acc1, run10_acc2 = model2_aileron(10, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
run11_acc1, run11_acc2 = model2_aileron(11, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)

# Accuracy
accuracy_DOF1 = [run1_acc1, run3_acc1, run8_acc1, run9_acc1, run10_acc1, run11_acc1]
accuracy_DOF2 = [run1_acc2, run3_acc2, run8_acc2, run9_acc2, run10_acc2, run11_acc2]
print(best_params)
# Plot accuracy
accuracy_plot(accuracy_DOF1, accuracy_DOF2)
