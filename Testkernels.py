import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from check_fulldata import dat_array

#Function to smooth data

def smooth_data(data, bandwidth=0.1):
    """Smooth the data using Gaussian kernel density estimation."""
    kde = gaussian_kde(data, bw_method=bandwidth)
    x = np.linspace(min(data), max(data), 1000)
    smoothed_data = kde(x)
    return x, smoothed_data

#Test with some data
DeltaDrum = dat_array(f"run1/aircraft/DeltaDrumAil")

x, smoothed_data = smooth_data(DeltaDrum, bandwidth=0.1)

plt.plot(x, smoothed_data)
plt.show()