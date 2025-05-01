import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from check_fulldata import dat_array

# Function to smooth data
def smooth_data(data):
    kde = gaussian_kde(data)  # Create a Gaussian KDE
    x = np.linspace(min(data), max(data), len(data))  # Generate points for evaluation
    smoothed_data = kde(x)  # Evaluate the KDE at these points
    return smoothed_data

# Test with some data
DeltaDrum = dat_array(f"run1/aircraft/DeltaDrumAil")
smoothed_data = smooth_data(DeltaDrum)

# Optional: Plot the smoothed data
t_values = np.linspace(0, len(DeltaDrum) - 1, len(DeltaDrum)) / 1000  # Original time array for plotting
plt.plot(t_values, smoothed_data, label="Smoothed Data")
plt.plot(t_values, DeltaDrum, label="Original Data", alpha=0.5)
plt.xlabel("Time (s)")
plt.ylabel("DeltaDrum")
plt.legend()
plt.grid()
plt.show()