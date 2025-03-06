def runge_kutta(f, y0, t0, tf, h):
    n = int((tf - t0) / h)
    t = t0
    y = y0
    for i in range(n):
        k1 = h * f(t, y)
        k2 = h * f(t + h / 2, y + k1 / 2)
        k3 = h * f(t + h / 2, y + k2 / 2)
        k4 = h * f(t + h, y + k3)
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += h
    return y

# Example usage:
# Define the differential equation dy/dt = f(t, y)
def f(t, y):
    return t - y

# Initial conditions
y0 = 1
t0 = 0
tf = 2
h = 0.1

# Solve the differential equation
result = runge_kutta(f, y0, t0, tf, h)
print("The solution at t =", tf, "is y =", result)