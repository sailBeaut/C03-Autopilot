import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


#Function to smooth data

def smooth_data(data, bandwidth=0.1):
    """Smooth the data using Gaussian kernel density estimation."""
    kde = gaussian_kde(data, bw_method=bandwidth)
    x = np.linspace(min(data), max(data), 1000)
    smoothed_data = kde(x)
    return x, smoothed_data

#Test with some data

data = np.random.normal(0, 1, 1000)  # Example data
x, smoothed_data = smooth_data(data, bandwidth=0.1)

plt.plot(x, smoothed_data)
plt.show()