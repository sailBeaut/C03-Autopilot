




















import numpy as np
import sympy as sp
    
# Define variables 
t = sp.symbols('t')
r1 = sp.symbols('r1')
r2 = sp.symbols('r2')
k1 = sp.symbols('k1')
k2 = sp.symbols('k2')
c1 = sp.symbols('c1')
c2 = sp.symbols('c2')
j1 = sp.symbols('j1')
j2 = sp.symbols('j2')
l  = sp.symbols('l')
#x1 = theta 1 and x2 = theta 2
x1, x2 = sp.Function('x1')(t), sp.Function('x2')(t)
T = sp.Function('T')(t)
# Define Martices and Vectors
#DAXF
D = sp.Matrix([x1.diff(t), x2.diff(t), x1.diff(t, t), x2.diff(t, t)])

A = sp.Matrix([[0, 0, 1, 0],[0, 0, 0, 1],[-k1*r1**2/j1, k1*r1*r2/j1, -c1*r1**2/j1,c1*r1*r2/j1],[k1*r1*r2/j2, (k1*r2**2 - k2*l**2)/j2, c1*r1*r2/j2, (c1*r2**2 - c2*l**2)/j2]])

X = sp.Matrix([x1, x2, x1.diff(t), x2.diff(t)])

F = sp.Matrix([0, 0, T/j1, 0])





def f(Y, t, M, C, K, F):
    n = len(K)
    x = Y[:n]
    v = Y[n:]
    
    dxdt = v
    dvdt = np.linalg.inv(M) @ (F - C @ v - K @ x)
    
    return np.hstack((dxdt, dvdt))

def runge_kutta4(Y0, t, M, C, K, F): 
    h = t[1] - t[0]
    Y = np.zeros((len(t), len(Y0)))
    Y[0, :] = Y0

    for i in range(len(t) - 1):
        k1 = f(Y[i, :], t[i], M, C, K, F)
        k2 = f(Y[i, :] + 0.5 * h * k1, t[i] + 0.5 * h, M, C, K, F)
        k3 = f(Y[i, :] + 0.5 * h * k2, t[i] + 0.5 * h, M, C, K, F)
        k4 = f(Y[i, :] + h * k3, t[i] + h, M, C, K, F)
        
        Y[i+1, :] = Y[i, :] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return Y

# Example parameters
n = 2  # 2-DOF system
M = np.array([[2, 0], [0, 1]])  # Mass matrix
C = np.array([[0.1, 0], [0, 0.2]])  # Damping matrix
K = np.array([[50, -10], [-10, 20]])  # Stiffness matrix
F = np.array([1, 0])  # External force

t = np.linspace(0, 10, 1000)  # Time vector
Y0 = np.zeros(2 * n)  # Initial conditions (zero displacement & velocity)

Y = runge_kutta4(Y0, t, M, C, K, F)

# Extract displacements and velocities
x1 = Y[:, 0]  # Displacement of DOF 1
x2 = Y[:, 1]  # Displacement of DOF 2




# Define the System
system = sp.simplify(A * X + F - D)

# Extract Equations
eqs = [sp.Eq(system[i], 0) for i in range(system.shape[0])]

# Solve the System
sol = [sp.dsolve(eq) for eq in eqs]

# Print Solutions
for s in sol:
	print(s)

''''
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
'''