import h5py
import sympy as sp
from check_data import dat_array, print_struc
import numpy as np
import matplotlib.pyplot as plt
from Functions_for_sys2 import model2, accuracy_plot_elev

# ...existing code...

# Parameters - Modified ranges and values
k1_range = range(1000, 50000, 1000)  # Smaller, more focused range
k2_numvalue = 15                   
c2_numvalue = 4.5                  
divfactor = 1.5                    
c1_numvalue = 350                  
k_g = 0.22
a_velo = 0.0000001

# On Or Off
extragraphs = False
showmainplots = True  # Set to True to visualize intermediate results
printeigenvalues = False

# Function to calculate average accuracy for DOF 2
def calculate_average_accuracy_dof2(accuracy_DOF2):
    if not accuracy_DOF2 or all(np.isnan(accuracy_DOF2)):
        return 0.0
    valid_values = [x for x in accuracy_DOF2 if not np.isnan(x)]
    return np.mean(valid_values) if valid_values else 0.0

# Optimization loop
best_accuracy = 0  # Start from 0 instead of 70
best_k1 = None
all_accuracies = []  # Track all valid accuracies

# Grid search over k1 range
for k1_numvalue in k1_range:
    try:
        # Run the model with current k1 value
        run1_acc1, run1_acc2 = model2(4, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
        run3_acc1, run3_acc2 = model2(5, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
        run8_acc1, run8_acc2 = model2(6, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
        run9_acc1, run9_acc2 = model2(7, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)

        # Elevator accuracy (DOF 2)
        accuracy_DOF2 = [run1_acc2, run3_acc2, run8_acc2, run9_acc2]
        
        # Print current values for debugging
        print(f"k1={k1_numvalue}, accuracies={accuracy_DOF2}")
        
        # Calculate average accuracy for DOF 2
        average_accuracy_dof2 = calculate_average_accuracy_dof2(accuracy_DOF2)
        
        if average_accuracy_dof2 > 0:  # Only consider non-zero accuracies
            all_accuracies.append((k1_numvalue, average_accuracy_dof2))
            
            # Update best if better than previous
            if average_accuracy_dof2 > best_accuracy:
                best_accuracy = average_accuracy_dof2
                best_k1 = k1_numvalue
                
    except Exception as e:
        print(f"Error with k1={k1_numvalue}: {str(e)}")
        continue

# Results
if best_k1 is not None:
    print("\nOptimization Results:")
    print(f"Best k1: {best_k1}")
    print(f"Best Elevator Accuracy: {best_accuracy:.2f}%")
    
    # Plot results if valid
    if all_accuracies:
        k1_values, accuracies = zip(*all_accuracies)
        plt.figure(figsize=(10, 6))
        plt.plot(k1_values, accuracies, 'b-')
        plt.xlabel('k1 values')
        plt.ylabel('Average Accuracy (%)')
        plt.title('Elevator Accuracy vs k1 Parameter')
        plt.grid(True)
        plt.show()
else:
    print("No valid parameters found. Try adjusting the parameter ranges or check the model2 function.")
