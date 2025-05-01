import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from check_data import dat_array, print_struc


DeltaDrum = dat_array("run1/aircraft/DeltaDrumAil")



data = DeltaDrum
time =  np.linspace(0, len(DeltaDrum) - 1, len(DeltaDrum)) / 1000  # Original time array for plotting
# Step 2: Apply Kernel Density Estimation (KDE) for Smoothing
kde = gaussian_kde(data, bw_method=0.1)
data_smoothed_kde = kde(data)  # Smoothed KDE values

# Step 3: Moving Average Smoothing (Alternative)
window_size = 13 # Adjust the window for more or less smoothing
data_smoothed_ma = np.convolve(data, np.ones(window_size) / window_size, mode='same')
  
# Step 4: Plot the Results
plt.figure(figsize=(10, 6))
plt.plot(time, data, label='Original Data', alpha=0.5)  # Original data
plt.plot(time, data_smoothed_kde, label='KDE Smoothed Data', color='orange')
plt.plot(time, data_smoothed_ma, label='Moving Average Smoothed Data', color='green')
plt.title('Smoothing Techniques Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()
