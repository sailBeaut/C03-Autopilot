import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Updated data
data = {
    "delta_ail_max": [-0.0525, 0.0552, -0.0575, 0.056, -0.0423, 0.054, -0.0438, 0.056, -0.0422, 0.046, -0.042, 0.046],
    "I_max": [0.855, -0.86, 0.86, -0.868, 0.848, -0.87, 0.858, -0.88, 0.86, -0.89, 0.864, -0.88],
    "q_avg": [1.18e5, 1.18e5, 1.46e5, 1.46e5, 1.46e5, 1.46e5, 1.46e5, 1.46e5, 1.84e5, 1.84e5, 1.82e5, 1.82e5]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[["I_max", "q_avg"]]
y = df["delta_ail_max"]

# Fit linear regression
model = LinearRegression()
model.fit(X, y)

# Coefficients
coeffs = model.coef_
intercept = model.intercept_

# Prediction function
def delta_ail_model(I, q):
    return coeffs[0] * I + coeffs[1] * q + intercept

# Create meshgrid for surface
I_vals = np.linspace(-0.9, 0.9, 50)
q_vals = np.linspace(1.1e5, 1.9e5, 50)
I_mesh, q_mesh = np.meshgrid(I_vals, q_vals)
delta_mesh = delta_ail_model(I_mesh, q_mesh)

# Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(I_mesh, q_mesh, delta_mesh, alpha=0.7, cmap='viridis')
ax.scatter(df["I_max"], df["q_avg"], df["delta_ail_max"], color='red', label='Data points')

ax.set_xlabel("Input Current (I)")
ax.set_ylabel("Dynamic Pressure (q) [Pa]")
ax.set_zlabel("Max Aileron Deflection (rad)")
ax.set_title("Max Aileron Deflection vs Input Current and Dynamic Pressure")
ax.legend()

plt.tight_layout()
plt.show()

# Print equation
print(f"Model equation:\nΔail_max = {coeffs[0]:.4f} * I + {coeffs[1]:.4e} * q + {intercept:.4f}")
