import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

c1list = [1.4135, 2.31, 1.4025, 0.8855, 0.935, 1.485]

k1list = [3.4045, 3.7345, 3.916, 3.7675, 4.3395, 4.4]

c2list = [0.0000004, 0.0000004, 0.0000003, 0.0000003, 0.0000002, 0.0000002]
V_knots = [40,40,60,60,80,80]

dlist = [4.455, 2.227, 4.455, 8.910, 8.910, 4.455]

darray = np.array(dlist).reshape(-1, 1)
c1array = np.array(c1list)
model = LinearRegression()
model.fit(darray, c1array)

# Predict values
c1_pred = model.predict(darray)

# Plot the regression
plt.scatter(dlist, c1list, label='Data')
plt.plot(dlist, c1_pred, color='red', label='Regression Line')
plt.xlabel('dlist')
plt.ylabel('c1list')
plt.legend()
plt.show()

