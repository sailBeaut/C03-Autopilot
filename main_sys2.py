import h5py
import sympy as sp
from check_data import dat_array, print_struc
import numpy as np
import matplotlib.pyplot as plt
from Functions_for_sys2 import model2_aileron, accuracy_plot

#Parameters
divfactor = 2
k1_numvalue = 380000
k2_numvalue = 12
c1_numvalue = 350
c2_numvalue = 4.5
k_g = 0.22
a_velo = 0.0000001

#On Or Off
extragraphs = False
showmainplots = False
printeigenvalues = False

#Run the model
run1_acc1, run1_acc2 =   model2_aileron(1, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
run3_acc1, run3_acc2 =   model2_aileron(3, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
run8_acc1, run8_acc2 =   model2_aileron(8, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
run9_acc1, run9_acc2 =   model2_aileron(9, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)    
run10_acc1, run10_acc2 = model2_aileron(10, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
run11_acc1, run11_acc2 = model2_aileron(11, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)

# Accuracy
accuracy_DOF1 = [run1_acc1, run3_acc1, run8_acc1, run9_acc1, run10_acc1, run11_acc1]
accuracy_DOF2 = [run1_acc2, run3_acc2, run8_acc2, run9_acc2, run10_acc2, run11_acc2]

# # # Plot accuracy

accuracy_plot(accuracy_DOF1, accuracy_DOF2)

