import sympy as sp
import h5py
from check_data import dat_array

print(dat_array("run1/aircraft/IservoAil"))

# Define time variable
t = sp.symbols('t')

# Define state vector and input
x1, x2 = sp.Function('x1')(t), sp.Function('x2')(t)
u = sp.Function('u')(t)

# Define system matrices
A = sp.Matrix([[0, 1], [-2, -3]])  # Example 2x2 system matrix
B = sp.Matrix([[0], [1]])		  # Example input matrix
X = sp.Matrix([x1, x2])			# State vector

# Define state-space equation
eqs = X.diff(t) - (A * X + B * u)

# Solve the system using dsolve
sol = sp.dsolve(eqs)

# Display solution
sol
