import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Data
data = {
    "delta_ail_max": [-0.0525, -0.0575, -0.0423, -0.0438, -0.0422, -0.042],
    "I_max": [0.855, 0.86, 0.848, 0.858, 0.86, 0.864],
    "q_avg": [1.18e5, 1.46e5, 1.46e5, 1.46e5, 1.84e5, 1.82e5]
}

df = pd.DataFrame(data)

# Add squared dynamic pressure term
df["q_sq"] = df["q_avg"]**2

# Features and target
X = df[["I_max", "q_avg", "q_sq"]]
y = df["delta_ail_max"]

# Fit polynomial regression model
model = LinearRegression()
model.fit(X, y)

# Coefficients
a, b, c = model.coef_
d = model.intercept_

# Print the equation
print(f"Model:\nΔail_max = {a:.4f} * I + {b:.4e} * q + {c:.4e} * q² + {d:.4f}")

# Generate prediction grid
I_vals = np.linspace(-0.9, 0.9, 50)
q_vals = np.linspace(1.1e5, 1.9e5, 50)
I_mesh, q_mesh = np.meshgrid(I_vals, q_vals)
q_sq_mesh = q_mesh ** 2

# Predict using the fitted model
delta_pred = a * I_mesh + b * q_mesh + c * q_sq_mesh + d

# Plot 3D surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(I_mesh, q_mesh, delta_pred, cmap='viridis', alpha=0.7)
ax.scatter(df["I_max"], df["q_avg"], df["delta_ail_max"], color='red', label='Data')
ax.set_xlabel("Input Current (I)")
ax.set_ylabel("Dynamic Pressure (q) [Pa]")
ax.set_zlabel("Max Aileron Deflection (rad)")
ax.set_title("Polynomial Model: Δail_max vs I and q")
ax.legend()
plt.tight_layout()
plt.show()
