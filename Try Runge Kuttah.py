import numpy as np
import matplotlib.pyplot as plt

# Define the ODE: dy/dx = f(x, y)
def f(x, y):
    return -2 * x * y  # Example: dy/dx = -2xy

# Runge-Kutta 4th order method
def runge_kutta_4(f, x0, y0, x_end, h):
    x_values = np.arange(x0, x_end + h, h)  # Array of x values
    y_values = np.zeros(len(x_values))  # Array to store y values
    y_values[0] = y0  # Initial condition

    for i in range(1, len(x_values)):
        x_n = x_values[i - 1]
        y_n = y_values[i - 1]

        k1 = h * f(x_n, y_n)
        k2 = h * f(x_n + h / 2, y_n + k1 / 2)
        k3 = h * f(x_n + h / 2, y_n + k2 / 2)
        k4 = h * f(x_n + h, y_n + k3)

        y_values[i] = y_n + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x_values, y_values

# Parameters
x0, y0 = 0, 1  # Initial condition y(0) = 1
x_end = 2  # Solve from x=0 to x=2
h = 0.1  # Step size

# Solve ODE
x_vals, y_vals = runge_kutta_4(f, x0, y0, x_end, h)

# Plot the results
plt.plot(x_vals, y_vals, 'b-', label="RK4 Approximation")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Runge-Kutta Method (RK4) for dy/dx = -2xy")
plt.legend()
plt.grid()
plt.show()
