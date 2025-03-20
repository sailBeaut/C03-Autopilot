import h5py
import sympy as sp
from check_data import dat_array, print_struc
import numpy as np
import matplotlib.pyplot as plt
from Functions_for_sys2 import model2_aileron, plot_histogram

#Parameters

k1_numvalue = 400000 
k2_numvalue = 10 
c1_numvalue = 250
c2_numvalue = 4.5
k_g = 0.22
a_velo = 1.225E-6
extragraphs = False
showmainplots = False
printeigenvalues = False

run1_acc1, run1_acc2 = model2_aileron(1, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
run3_acc1, run3_acc2 = model2_aileron(3, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
run8_acc1, run8_acc2 = model2_aileron(8, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
run9_acc1, run9_acc2 = model2_aileron(9, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)    
run10_acc1, run10_acc2 = model2_aileron(10, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
run11_acc1, run11_acc2 = model2_aileron(11, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)

