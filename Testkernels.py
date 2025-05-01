import numpy as np
from scipy.ndimage import gaussian_filter1d

def smooth_data(data, sigma=2):
    """
    Smooth the input data using a Gaussian filter.

    Parameters:
    - data (array-like): The input data to be smoothed.
    - sigma (float): The standard deviation for the Gaussian kernel.

    Returns:
    - data_smoothed (ndarray): The smoothed data as an array.
    """
    data_smoothed = gaussian_filter1d(data, sigma=sigma)
    return data_smoothed