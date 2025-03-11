import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from check_data import dat_array, print_struc


IservoAil = dat_array("run1/aircraft/IservoAil")



current = IservoAil
time = np.linspace(0, 7000, 7001)
# Step 2: Apply Kernel Density Estimation (KDE) for Smoothing
kde = gaussian_kde(current, bw_method=0.1)
current_smoothed_kde = kde(current)  # Smoothed KDE values

# Step 3: Moving Average Smoothing (Alternative)
window_size = 50  # Adjust the window for more or less smoothing
current_smoothed_ma = np.convolve(current, np.ones(window_size) / window_size, mode='same')

# Step 4: Plot the Results
plt.figure(figsize=(12, 6))
plt.plot(time, current, alpha=0.5, label="Original Current (with Spikes)", color='gray')
plt.plot(time, current_smoothed_ma, label="Smoothed (Moving Average)", color='red', linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.title("Smoothing Input Current Data")
plt.legend()
plt.show()

