import h5py
import sympy as sp
from check_data import dat_array, print_struc
import numpy as np
import matplotlib.pyplot as plt
from Functions_for_sys2 import model2, accuracy_plot_elev

# Parameters
k1_range = range(0, 50001, 10000)  # Range for k1_numvalue
k2_range = range(10, 21, 2)        # Range for k2_numvalue
c2_range = np.arange(4.0, 5.1, 0.2)  # Range for c2_numvalue
divfactor_range = np.arange(1.0, 2.1, 0.1)  # Range for divfactor
c1_numvalue = 350  # c1 is kept constant
k_g = 0.22
a_velo = 0.0000001

# On Or Off
extragraphs = False
showmainplots = False
printeigenvalues = False

# Function to calculate average accuracy for DOF 2
def calculate_average_accuracy_dof2(accuracy_DOF2):
    return np.mean(accuracy_DOF2)

# Optimization loop
best_accuracy = 70
best_params = None

# Grid search over parameter ranges
for k1_numvalue in k1_range:
    for k2_numvalue in k2_range:
        for c2_numvalue in c2_range:
            for divfactor in divfactor_range:
                # Run the model with current parameters
                run1_acc1, run1_acc2 = model2(4, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
                run3_acc1, run3_acc2 = model2(5, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
                run8_acc1, run8_acc2 = model2(6, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
                run9_acc1, run9_acc2 = model2(7, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)    
                run10_acc1, run10_acc2 = model2(12, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
                run11_acc1, run11_acc2 = model2(13, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)

                # Accuracy
                accuracy_DOF1 = [run1_acc1, run3_acc1, run8_acc1, run9_acc1, run10_acc1, run11_acc1]
                accuracy_DOF2 = [run1_acc2, run3_acc2, run8_acc2, run9_acc2, run10_acc2, run11_acc2]

                # Calculate average accuracy for DOF 2
                average_accuracy_dof2 = calculate_average_accuracy_dof2(accuracy_DOF2)

                # Check if the current parameters give better accuracy for DOF 2
                if average_accuracy_dof2 > best_accuracy:
                    best_accuracy = average_accuracy_dof2
                    best_params = (k1_numvalue, k2_numvalue, c2_numvalue, divfactor)

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
print("Best Parameters:", best_params)
print("Best Accuracy:", best_accuracy)

# Plot accuracy
accuracy_plot_elev(accuracy_DOF1, accuracy_DOF2)
