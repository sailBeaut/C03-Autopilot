import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Open the HDF5 file in read mode
#file_path = "/Users/lennarthubbers/Desktop/processed-20250217_151129.hdf5"  # Replace with your actual file path
file_path = "filtered_data.hdf5"  # Replace with your actual file path



with h5py.File(file_path, "r") as f:
    def print_structure(name, obj):
        print(name, "->", "Group" if isinstance(obj, h5py.Group) else "Dataset")
    f.visititems(print_structure)
    Current_Data = f["run1/aircraft/IservoAil"][()]
    Time = [i  * 0.001 for i in range(0,len(Current_Data))]


import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def gaussian_process_regression(x_train, y_train, x_pred=None, length_scale=1.0, noise_level=0.1):
    """
    Performs Gaussian Process Regression (GPR) on given data.

    Parameters:
    - x_train (array): Known x values (1D array).
    - y_train (array): Corresponding y values.
    - x_pred (array, optional): X values for prediction. Defaults to 100 points in the range of x_train.
    - length_scale (float): Controls smoothness of the GP fit.
    - noise_level (float): Adds white noise for robustness.

    Returns:
    - x_pred (array): X values used for prediction.
    - y_pred (array): Predicted Y values.
    - sigma (array): Standard deviation (error estimation).
    - gp_model (GaussianProcessRegressor): Trained GP model (can be reused).
    """

    # Reshape x for scikit-learn
    x_train = np.array(x_train).reshape(-1, 1)
    
    # Define kernel (RBF for smoothness + WhiteKernel for noise handling)
    kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
    
    # Train the Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(x_train, y_train)

    # Define prediction points if not provided
    if x_pred is None:
        x_pred = np.linspace(min(x_train), max(x_train), 100).reshape(-1, 1)
    else:
        x_pred = np.array(x_pred).reshape(-1, 1)

    # Predict with uncertainty
    y_pred, sigma = gp.predict(x_pred, return_std=True)

    # Return results
    return x_pred.flatten(), y_pred, sigma, gp

# Example Data (Irregularly Spaced)
x_data = np.array([0, 1, 2.5, 3, 4.7, 6, 8, 9.5, 10])
y_data = np.sin(x_data) + np.random.normal(0, 0.1, len(x_data))  # Noisy sine wave

# Call function for Gaussian Process Regression
x_pred, y_pred, sigma, gp_model = gaussian_process_regression(x_data, y_data)

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(x_data, y_data, 'ro', label="Data Points")  # Original Data
plt.plot(x_pred, y_pred, 'b-', label="GP Prediction")  # GP Fit
plt.fill_between(x_pred, y_pred - sigma, y_pred + sigma, color='b', alpha=0.2, label="Confidence Interval")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Gaussian Process Regression with Uncertainty Estimation")
plt.show()