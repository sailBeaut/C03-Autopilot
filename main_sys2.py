import h5py
import sympy as sp
from check_fulldata import dat_array, print_struc
import numpy as np
import matplotlib.pyplot as plt
from Functions_for_sys2 import model2, accuracy_plot_ail, accuracy_plot_elev

#Parameters Aileron
divfactor = 2
k1_numvalue_ail = 400000
k2_numvalue_ail = 10.8
c1_numvalue_ail = 350
c2_numvalue_ail = 4.5
k_g = 0.22
a_velo_ail = 0.0000001

#Parameters Elevator
divfactor = 2
k1_numvalue_elev = 400000
k2_numvalue_elev = 10.8
c1_numvalue_elev = 350
c2_numvalue_elev = 4.5
k_g = 0.22
a_velo_elev = 0.0000001

#On Or Off
aileron = True
elevator = False
extragraphs = False
showmainplots = False
printeigenvalues = False

#Define Accuracy lists
accuracy_DOF1_ail = []
accuracy_DOF2_ail = []
accuracy_DOF1_elev = []
accuracy_DOF2_elev = []

#Run the model
for i in range(1, 14):
    if i == 2:
        print('There is no data for this case, run 2 is skipped')
        continue
    else:
        run = i
        if run in (1, 3, 8, 9, 10, 11) and aileron == True:
            acc_run_DOF1, acc_run_DOF2 = model2(run, divfactor, k1_numvalue_ail, k2_numvalue_ail, c1_numvalue_ail, c2_numvalue_ail, k_g, a_velo_ail, extragraphs, showmainplots, printeigenvalues)
            # Calculate accuracy for aileron
            accuracy_DOF1_ail.append(acc_run_DOF1)
            accuracy_DOF2_ail.append(acc_run_DOF2)
        elif run in (4, 5, 6, 7, 12, 13) and elevator == True:
            acc_run_DOF1, acc_run_DOF2 = model2(run, divfactor, k1_numvalue_elev, k2_numvalue_elev, c1_numvalue_elev, c2_numvalue_elev, k_g, a_velo_elev, extragraphs, showmainplots, printeigenvalues)
            # Calculate accuracy for elevator
            accuracy_DOF1_elev.append(acc_run_DOF1)
            accuracy_DOF2_elev.append(acc_run_DOF2)


# Plot accuracy
accuracy_plot_ail(accuracy_DOF1_ail, accuracy_DOF2_ail)
accuracy_plot_elev(accuracy_DOF1_elev, accuracy_DOF2_elev)