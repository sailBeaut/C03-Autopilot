from sys2_updated import *
import h5py
import sympy as sp
from check_data import dat_array, print_struc
import numpy as np
import matplotlib.pyplot as plt
from Testkernels import current_smoothed_ma


absolute_error1 = np.abs(Y_sol[:, 0] - DeltaDrumAil)
absolute_error2 = np.abs(-Y_sol[:, 1] - DeltaAil)

# Compute accuracy as percentage
error_norm1 = np.linalg.norm(absolute_error1) / np.linalg.norm(DeltaDrumAil)
accuracy1 = (1 - error_norm1) * 100
error_norm2 = np.linalg.norm(absolute_error2) / np.linalg.norm(DeltaAil)
accuracy2 = (1 - error_norm2) * 100

# Print accuracy
print(f"Model Accuracy of DOF1: {accuracy1:.2f}%")
print(f"Model Accuracy of DOF2: {accuracy2:.2f}%")