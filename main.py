import h5py
import sympy as sp
from check_data import dat_array, print_struc
import numpy as np
import matplotlib.pyplot as plt


DeltaAil = dat_array("run1/aircraft/DeltaAil")
IservoAil = dat_array("run1/aircraft/IservoAil")

# Parametric constants
c1 = 1.0  # damper constant TUNING PARAMETER
k1 = 1.0  # spring constant TUNING PARAMETER
kg = 1.0  # gain SET PARAMETER
Ie = 1.0  # moment of inertia TUNING PARAMETER

# State equation
A = sp.MatrixSymbol('A', 2, 2)  # 2x2 symbolic matrix
B = sp.MatrixSymbol('B', 2, 1)  # 2x1 symbolic matrix
x = sp.MatrixSymbol('x', 2, 1)  # 2x1 symbolic vector
u = sp.MatrixSymbol('u', 1, 1)  # Scalar input as a 1x1 matrix


A_val = sp.Matrix([[-(c1/Ie), -(k1/Ie)], [1, 0]])
B_val = sp.Matrix([[kg/Ie], [0]])
x_val = sp.Matrix([[DeltaAil[0]], [0]])


xdot = A @ x + B @ u
xinst = 0
xlist = []
vinst = 0

for i in range(7001):
	u_val = sp.Matrix([[IservoAil[i]]])
	if i !=0: 
		x_val = sp.Matrix([[DeltaAil[i]], [vinst]])
	xdot_evaluated = xdot.subs({A: A_val, B: B_val, x: x_val, u: u_val})
	xdot_result = xdot_evaluated.doit()
	if i%100==0:
		print(i)
	xinst += xdot_result[1]*0.001
	vinst += xdot_result[0]*0.001
	xlist.append(xinst)

plt.plot(np.linspace(0,7000,7001), xlist, color="r")
plt.plot(np.linspace(0,7000,7001), DeltaAil, color="b")
plt.show()

# Simplify and compute the result



