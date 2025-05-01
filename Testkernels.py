import numpy as np
import matplotlib.pyplot as plt
from check_data import dat_array, print_struc




def smooth_data(data, window_size = 13):
    data_smoothed_ma = np.convolve(data, np.ones(window_size) / window_size, mode='same')
    return data_smoothed_ma
