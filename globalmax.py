from scipy.optimize import differential_evolution
import numpy as np
from Functions_for_sys2 import model2

# --- Set your constants here ---
array = False
resolution = 2
flip = 1
divfactor = 1
k_g = 0.22
k1_numvalue = 500000
c1_numvalue = 50
clutch = 0

extragraphs = False
showmainplots = False
printeigenvalues = False

def objective(params):
    k2_numvalue, c2_numvalue, a_velo = params
    # Run your model with these parameters (c1 stays constant)
    _, run1_acc2 = model2(4, array, resolution, clutch, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
    _, run3_acc2 = model2(5, array, resolution, clutch, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
    _, run8_acc2 = model2(6, array, resolution, clutch, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
    _, run9_acc2 = model2(7, array, resolution, clutch, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
    _, run10_acc2 = model2(12, array, resolution, clutch, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
    _, run11_acc2 = model2(13, array, resolution, clutch, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues)
    accuracy_DOF2 = [run1_acc2, run3_acc2, run8_acc2, run9_acc2, run10_acc2, run11_acc2]
    return -np.mean(accuracy_DOF2)  # Negative for maximization

# Bounds for k1, k2, c2
bounds = [
    (10, 40),           # k2
    (1, 20),            # c2
    (1e-07, 9e-07)      #a_velo       
]

result = differential_evolution(objective, bounds, disp=True)
print("Best parameters:", result.x)
print("Best accuracy for DOF2:", -result.fun)