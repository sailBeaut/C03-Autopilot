import sympy as sp

# Define variables
t = sp.symbols('t')
x1, x2 = sp.Function('x1')(t), sp.Function('x2')(t)

# Define system parameters
m1, m2 = sp.symbols('m1 m2')  # Masses
c1, c2 = sp.symbols('c1 c2')  # Damping
k1, k2 = sp.symbols('k1 k2')  # Stiffness

# Define external forces (non-homogeneous part)
F1, F2 = sp.Function('F1')(t), sp.Function('F2')(t)

# Define second-order equations
eq1 = sp.Eq(m1 * x1.diff(t, t) + c1 * x1.diff(t) + k1 * x1, F1)
eq2 = sp.Eq(m2 * x2.diff(t, t) + c2 * x2.diff(t) + k2 * x2, F2)

# Convert to first-order system
v1, v2 = sp.Function('v1')(t), sp.Function('v2')(t)  # Define velocity variables

eq1_first = sp.Eq(x1.diff(t), v1)
eq2_first = sp.Eq(x2.diff(t), v2)

eq3_first = sp.Eq(v1.diff(t), (F1 - c1 * v1 - k1 * x1) / m1)
eq4_first = sp.Eq(v2.diff(t), (F2 - c2 * v2 - k2 * x2) / m2)

# Solve the system
sol = sp.dsolve([eq1_first, eq2_first, eq3_first, eq4_first])

# Display solutions
for s in sol:
    print(s)

