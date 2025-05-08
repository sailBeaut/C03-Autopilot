import h5py
import sympy as sp
from check_data import dat_array, print_struc
import numpy as np
import matplotlib.pyplot as plt
from Functions_for_sys2 import model2, accuracy_plot_elev

# Parameters
k1_numvalue = 500000
k2_numvalue = 27.5
c1_numvalue = 0  
c2_numvalue = 4.5
k_g = 0.22
a_velo = 5e-7  
divfactor = 1
flip = 1
resolution = 1

# On Or Off
array = False
extragraphs = False
showmainplots = False
printeigenvalues = False

# Function to calculate average accuracy for DOF 2
def calculate_average_accuracy_dof2(accuracy_DOF2):
    return np.mean(accuracy_DOF2)

# Optimization loop
best_accuracy = 0
best_params = (k1_numvalue, k2_numvalue, c2_numvalue)

while best_accuracy < 65:
    # Run the model with current parameters
    run1_acc1, run1_acc2 = model2(4, array, resolution, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
    run3_acc1, run3_acc2 = model2(5, array, resolution, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
    run8_acc1, run8_acc2 = model2(6, array, resolution, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
    run9_acc1, run9_acc2 = model2(7, array, resolution, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
    run10_acc1, run10_acc2 = model2(12, array, resolution, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
    run11_acc1, run11_acc2 = model2(13, array, resolution, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)

    # Accuracy
    accuracy_DOF2 = [run1_acc2, run3_acc2, run8_acc2, run9_acc2, run10_acc2, run11_acc2]

    # Calculate average accuracy for DOF 2
    average_accuracy_dof2 = calculate_average_accuracy_dof2(accuracy_DOF2)

    # Update parameters for next iteration
    #k1_numvalue += 20000  
    #k2_numvalue += 2     
    #c1_numvalue += 0.0000000001   
    #c2_numvalue += 0.5    
    #a_velo += 0.00000005

    print(f"Best Parameters: k1={k1_numvalue}, c1 = {c1_numvalue} , k2={k2_numvalue}, c2={c2_numvalue}, a_velo={a_velo}")
    print(f"Average Accuracy for DOF 2: {average_accuracy_dof2:.2f}%")

# Use the best parameters found
k1_numvalue, k2_numvalue, c2_numvalue = best_params

# Final run with best parameters
run1_acc1, run1_acc2 = model2(4, array, resolution, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
run3_acc1, run3_acc2 = model2(5, array, resolution, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
run8_acc1, run8_acc2 = model2(6, array, resolution, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
run9_acc1, run9_acc2 = model2(7, array, resolution, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
run10_acc1, run10_acc2 = model2(12, array, resolution, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
run11_acc1, run11_acc2 = model2(13, array, resolution, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)

# Accuracy
accuracy_DOF2 = [run1_acc2, run3_acc2, run8_acc2, run9_acc2, run10_acc2, run11_acc2]
print(f"Best Parameters: {best_params}")
print(f"Best Accuracy for DOF 2: {best_accuracy:.2f}%")

# Plot accuracy for DOF 2
accuracy_plot_elev([], accuracy_DOF2)