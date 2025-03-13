import h5py
import sympy as sp
from check_data import dat_array, print_struc
import numpy as np
import matplotlib.pyplot as plt
from Testkernels import current_smoothed_ma

DeltaAil = dat_array("run1/aircraft/DeltaAil")
DeltaDrumAil = dat_array("run1/aircraft/DeltaDrumAil")
IservoAil = dat_array("run1/aircraft/IservoAil")
# Gain
k_g = -0.45

# Aileron
# Step 1: Define symbolic variables for Mass (M), Damping (C), and Stiffness (K)
j1, j2 = sp.symbols('j1 j2')  # Masses
k1, k2 = sp.symbols('k1 k2')  # Stiffness values
c1, c2 = sp.symbols('c1 c2')  # Damping coefficients
t = sp.symbols('t')
r1 = sp.symbols('r1')
r2 = sp.symbols('r2')
l = sp.symbols('l')

# Define mass, stiffness, and damping matrices
M = sp.Matrix([[j1, 0], [0, j2]])  # Mass matrix
C = sp.Matrix([[c1 * r1 ** 2, -c1 * r1 * r2], [-c1 * r1 * r2, c2 + c1 * r2 ** 2]])  # Damping matrix
K = sp.Matrix([[k1 * r1 ** 2, -k1 * r1 * r2], [-k1 * r1 * r2, k2 + k1 * r2 ** 2]])  # Stiffness matrix

# Define force vector (external forces)
F = sp.Matrix([sp.sin(t), 0])  # Example force applied to first DOF
# Assume input Torque is sinus function

# Compute M inverse
M_inv = M.inv()

# Step 2: Define first-order system
x = sp.Matrix(sp.symbols('x1 x2'))  # Displacement vector
v = sp.Matrix(sp.symbols('v1 v2'))  # Velocity vector

# Combine into state-space form
Y = sp.Matrix.vstack(x, v)

# Convert symbolic to numerical
subs_dict = {j1: 5.4E-5, j2: 7.97E-2, k1: 1E5, k2: 0.5E5, c1: 2E2, c2: 1E2, r1: 2.52E-2, r2: 7.9E-2}
M_num = np.array(M.subs(subs_dict)).astype(np.float64)
K_num = np.array(K.subs(subs_dict)).astype(np.float64)
C_num = np.array(C.subs(subs_dict)).astype(np.float64)

# Calculate eigenvalues
j1_value, j2_value, k1_value, k2_value, c1_value, c2_value, r1_value, r2_value = subs_dict[j1], subs_dict[j2], \
subs_dict[k1], subs_dict[k2], subs_dict[c1], subs_dict[c2], subs_dict[r1], subs_dict[r2]

def eigenvalues(j1_value, j2_value, k1_value, k2_value, c1_value, c2_value, r1_value, r2_value):
    j1, j2, k1, k2, c1, c2, r1, r2 = j1_value, j2_value, k1_value, k2_value, c1_value, c2_value, r1_value, r2_value
    A = np.array(
        [[0, 0, 1, 0], [0, 0, 0, 1], [-k1 * r1 ** 2 / j1, k1 * r1 * r2 / j1, -c1 * r1 ** 2 / j1, c1 * r1 * r2 / j1],
         [k1 * r1 * r2 / j2, -(k2 + k1 * r2 ** 2) / j2, c1 * r1 * r2 / j2, -(c2 + c1 * r2 ** 2) / j2]])

    # Calculate the eigenvalues
    eigenvalues = np.linalg.eigvals(A)

    # Print the eigenvalues
    print("Eigenvalues of the matrix are:", eigenvalues)


# Convert system equations to NumPy function
def system(Y, t):
    x = Y[:2]  # First two elements are displacements
    v = Y[2:]  # Last two elements are velocities
    F_num = np.array([np.interp(t, t_values, current_smoothed_ma) * k_g, 0])  # Numerical force

    dxdt = v
    dvdt = np.linalg.inv(M_num) @ (F_num - C_num @ v - K_num @ x)

    return np.hstack((dxdt, dvdt))


# Step 3: Implement RK4 for numerical integration
def runge_kutta4(f, Y0, t):
    n = len(t)
    h = t[1] - t[0]  # Time step
    Y = np.zeros((n, len(Y0)))  # Store results
    Y[0] = Y0

    for i in range(n - 1):
        k1 = f(Y[i], t[i])
        k2 = f(Y[i] + 0.5 * h * k1, t[i] + 0.5 * h)
        k3 = f(Y[i] + 0.5 * h * k2, t[i] + 0.5 * h)
        k4 = f(Y[i] + h * k3, t[i] + h)

        Y[i + 1] = Y[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return Y


# Time settings
t_values = np.linspace(0, 7, 7001)  # Time from 0 to 7 sec
Y0 = [0, 0, 0, 0]  # Initial conditions: x1 = x2 = v1 = v2 = 0

# Solve using RK4
Y_sol = runge_kutta4(system, Y0, t_values)

print(t_values)
eigenvalues(j1_value, j2_value, k1_value, k2_value, c1_value, c2_value, r1_value, r2_value)

# Step 4: Plot results
#plt.plot(t_values, Y_sol[:, 0] * (180 * np.pi), label="x1 (DOF 1)")
plt.subplot(2, 2, 1)
plt.plot(t_values, Y_sol[:, 1], label="x2 (DOF 2)")
plt.plot(t_values, DeltaAil, label="DeltaAil")
plt.xlabel("Time (s)")
plt.ylabel("Displacement of DOF 2")
plt.title("MDOF System Response (RK4)")
plt.legend()
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(t_values, Y_sol[:, 1], label="x2 (DOF 2)")
plt.xlabel("Time (s)")
plt.ylabel("Displacement of DOF 2")
plt.title("MDOF System Response (RK4)")
plt.legend()
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(t_values, -Y_sol[:, 0], label="x1 (DOF 1)")
plt.xlabel("Time (s)")
plt.ylabel("Displacement of DOF 1")
plt.title("MDOF System Response (RK4)")
plt.legend()
plt.grid()


plt.subplot(2, 2, 4)
plt.plot(t_values, DeltaDrumAil, label="DeltaDrumAil")
plt.xlabel("Time (s)")
plt.ylabel("Displacement of DOF 1")
plt.title("MDOF System Response (RK4)")
plt.legend()
plt.grid()










plt.show()


plt.plot(t_values, Y_sol[:, 0]*(180*np.pi), label="x1 (DOF 1)")
plt.plot(t_values, Y_sol[:, 1]*(180*np.pi), label="x2 (DOF 2)")
plt.xlabel("Time (s)")
plt.ylabel("Displacement")
plt.title("MDOF System Response (RK4)")
plt.legend()
plt.grid()
plt.show()
