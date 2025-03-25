import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Given data
V_knots = np.array([40, 40, 60, 60, 80, 80])
dlist = np.array([4.455, 2.227, 4.455, 8.910, 8.910, 4.455])
c1list = np.array([1.4135, 2.31, 1.4025, 0.8855, 0.935, 1.485])
'''

# Given data
V_knots = np.array([40, 40, 60, 60, 80, 80])
dlist = np.array([4.455, 2.227, 4.455, 8.910, 8.910, 4.455])
k1list = np.array([3.4045, 3.7345, 3.916, 3.7675, 4.3395, 4.4])

# Fit a linear regression model
X = np.column_stack((V_knots, dlist))
y = k1list
model = LinearRegression().fit(X, y)

# Create a mesh grid for the hyperplane
V_range = np.linspace(min(V_knots), max(V_knots), 10)
d_range = np.linspace(min(dlist), max(dlist), 10)
V_grid, d_grid = np.meshgrid(V_range, d_range)
k1_grid = model.predict(np.column_stack((V_grid.ravel(), d_grid.ravel()))).reshape(V_grid.shape)

# Create 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(V_knots, dlist, k1list, c='b', marker='o', label='Data points')

# Plot the hyperplane
ax.plot_surface(V_grid, d_grid, k1_grid, color='r', alpha=0.5)

# Labels and title
ax.set_xlabel('V_knots')
ax.set_ylabel('d')
ax.set_zlabel('k1')
ax.set_title('3D Scatter Plot with Regression Plane')
ax.legend()

# Show plot
plt.show()
'''
'''

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(dlist.reshape(-1, 1))
model = LinearRegression().fit(X_poly, c1list)

# Create a range of d values for prediction
d_range = np.linspace(min(dlist), max(dlist), 100).reshape(-1, 1)
d_range_poly = poly.transform(d_range)
c1_pred = model.predict(d_range_poly)

# Create figure
plt.figure(figsize=(8, 6))
plt.scatter(dlist, c1list, color='b', label='Data points')
plt.plot(d_range, c1_pred, color='r', label='Quadratic Fit')

# Labels and title
plt.xlabel('d')
plt.ylabel('c1')
plt.title('Quadratic Regression of c1 with d')
plt.legend()

# Show plot
plt.show()
'''

X = dlist.reshape(-1, 1)
y = c1list
model = LinearRegression().fit(X, y)

# Create a range of d values for prediction
d_range = np.linspace(min(dlist), max(dlist), 100).reshape(-1, 1)
c1_pred = model.predict(d_range)

# Create figure
plt.figure(figsize=(8, 6))
plt.scatter(dlist, c1list, color='b', label='Data points')
plt.plot(d_range, c1_pred, color='r', label='Linear Fit')

# Labels and title
plt.xlabel('d')
plt.ylabel('c1')
plt.title('Linear Regression of c1 with d')
plt.legend()

# Show plot
plt.show()
