import h5py
import sympy as sp
from check_data import dat_array, print_struc
import numpy as np
import matplotlib.pyplot as plt
from Functions_for_sys2 import model2, accuracy_plot_elev

# Parameters
k1_numvalue = 500000
k2_numvalue = 31
c1_numvalue = 200  # c1 is kept constant
c2_numvalue = 6
k_g = 0.22
a_velo = 0.0000001
divfactor = 4
flip = 1
resolution = 2

# On Or Off
extragraphs = False
showmainplots = False
printeigenvalues = False

# Function to calculate average accuracy for DOF 2
def calculate_average_accuracy_dof2(accuracy_DOF2):
    return np.mean(accuracy_DOF2)

# Optimization loop
best_accuracy = 0
best_params = (k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue)


while best_accuracy < 70:
    # Run the model with current parameters
    run1_acc1, run1_acc2 = model2(4,resolution, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
    run3_acc1, run3_acc2 = model2(5, resolution, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
    run8_acc1, run8_acc2 = model2(6, resolution, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
    run9_acc1, run9_acc2 = model2(7, resolution, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)    
    run10_acc1, run10_acc2 = model2(12, resolution, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
    run11_acc1, run11_acc2 = model2(13, resolution, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)

    print(c1_numvalue)

    # Accuracy
    accuracy_DOF1 = [run1_acc1, run3_acc1, run8_acc1, run9_acc1, run10_acc1, run11_acc1]
    accuracy_DOF2 = [run1_acc2, run3_acc2, run8_acc2, run9_acc2, run10_acc2, run11_acc2]

    # Calculate average accuracy for DOF 2
    average_accuracy_dof2 = calculate_average_accuracy_dof2(accuracy_DOF2)
    average_acc= print(np.mean(accuracy_DOF2))
    # Check if the current parameters give better accuracy for DOF 2
    if average_accuracy_dof2 > best_accuracy:
        best_accuracy = average_accuracy_dof2
        best_params = (k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, divfactor)

    # Update parameters for next iteration (example: increment k1, k2, c2)
    #k1_numvalue += 100000
    #k2_numvalue += 1
    c1_numvalue += 50
    #c2_numvalue += 1
    #divfactor += 0.1

# Use the best parameters found
k1_numvalue, k2_numvalue, c2_numvalue, divfactor = best_params

# Final run with best parameters
run1_acc1, run1_acc2 = model2(4, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
run3_acc1, run3_acc2 = model2(5, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
run8_acc1, run8_acc2 = model2(6, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
run9_acc1, run9_acc2 = model2(7, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)    
run10_acc1, run10_acc2 = model2(12, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
run11_acc1, run11_acc2 = model2(13, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)

# Accuracy
accuracy_DOF1 = [run1_acc1, run3_acc1, run8_acc1, run9_acc1, run10_acc1, run11_acc1]
accuracy_DOF2 = [run1_acc2, run3_acc2, run8_acc2, run9_acc2, run10_acc2, run11_acc2]
print(best_params)
# Plot accuracy
accuracy_plot_elev(accuracy_DOF1, accuracy_DOF2)