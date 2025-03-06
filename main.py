import h5py
import sympy as sp
from check_data import dat_array, print_struc

DeltaAil = dat_array("run1/aircraft/DeltaAil")
IservoAil = dat_array("run1/aircraft/IservoAil")

# Define symbolic variables
DeltaAil = sp.Symbol("DeltaAil")
DeltaAil_dot = sp.Symbol("DeltaAil_dot")
IservoAil = sp.Symbol("IservoAil")

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
x_val = sp.Matrix([[DeltaAil_dot], [DeltaAil]])
u_val = sp.Matrix([[IservoAil]])


xdot = A @ x + B @ u

xdot_evaluated = xdot.subs({A: A_val, B: B_val, x: x_val, u: u_val})

# Simplify and compute the result
xdot_result = xdot_evaluated.doit()

print(xdot_result)