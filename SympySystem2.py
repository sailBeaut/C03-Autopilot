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

#Define Equation:

eq1 = sp.Eq(A*X + F, D)

#Solve

sol = sp.dsolve(eq1)

#Print Sol

print(sol)