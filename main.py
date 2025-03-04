import sympy as sp
import h5py
from check_data import dat_array, print_struc

print_struc()
print(dat_array("run1/aircraft/IservoAil"))

# Define time variable
t = sp.symbols('t')


# Define state vector and input
delta_e_dot, delta_e = sp.Function('d_e_d')(t), sp.Function('d_e')(t)
i = sp.Function('i')(t)

# Define variables for system matrix
c1 = 1 # Damping coefficient
k1 = 1 # Spring constant
kg = 1 # Gain of the system
Ie = 1 # Moment of inertia of the elevator


# Define system matrices
A = sp.Matrix([[-(c1/Ie), -(k1/Ie)], [1, 0]])  # Example 2x2 system matrix
B = sp.Matrix([[kg/Ie], [0]])		  # Example input matrix
X = sp.Matrix([delta_e_dot, delta_e])			# State vector

# Define state-space equation
eqs = X.diff(t) - (A * X + B *i)

# Solve the system using dsolve
sol = sp.dsolve(eqs)

# Display solution
sol
