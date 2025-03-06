import h5py
import sympy as sp

with h5py.File("filtered_data.hdf5", "r") as f:
    DeltaAil = list(f["run1/aircraft/DeltaAil"])  # Fetch as list (length = 7001)
    IservoAil = list(f["run1/aircraft/IservoAil"])  # Fetch as list (length = 7001)
    # print("Data:", data)


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
A = sp.Matrix([[-(c1/Ie), -(k1/Ie)], [1, 0]])
B = sp.Matrix([[kg/Ie], [0]])
x = sp.Matrix([[DeltaAil_dot], [DeltaAil]])
u = sp.Matrix([[IservoAil], [0]])

def state_equation():
    xdot = A * x + B * u
