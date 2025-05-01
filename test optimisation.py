# import numpy as np
# from check_data import dat_array as load_data

# # Define search ranges
# c1_range = np.linspace(1.08, 1.12, 10)
# k1_range = np.linspace(9.0, 9.25, 10)
# Ie_range = np.linspace(0.018, 0.022, 10)

# best_accuracy = -np.inf
# best_params = ()

# combo_num = 0
# total_combos = len(c1_range) * len(k1_range) * len(Ie_range)

# # Grid search
# for c1 in c1_range:
#     for k1 in k1_range:
#         for Ie in Ie_range:
#             combo_num += 1
#             if combo_num % 50 == 0:
#                 print(f"Testing combo {combo_num} of {total_combos}")
#             Accuracy = []
#             for run_nr in (4, 5, 6, 7, 12, 13):
#                 DeltaElev = load_data(f"run{run_nr}/aircraft/DeltElev")
#                 IservoElev = load_data(f"run{run_nr}/aircraft/IservoElev")

#                 kg = -0.22  # Gain

#                 # System matrices
#                 A = np.array([[-(c1/Ie), -(k1/Ie)], [1, 0]])
#                 B = np.array([[-kg/Ie], [0]])

#                 x = np.array([[0], [0]])  # Initial state
#                 xlist = []

#                 dt = 0.001
#                 for i in range(7001):
#                     u = np.array([[IservoElev[i]]])
#                     xdot = A @ x + B @ u
#                     x = x + xdot * dt
#                     xlist.append(x[1, 0])

#                 xlist = np.array(xlist)
#                 absolute_error = np.abs(xlist - DeltaElev)
#                 error_norm = np.linalg.norm(absolute_error) / np.linalg.norm(DeltaElev)
#                 accuracy = (1 - error_norm) * 100
#                 Accuracy.append(accuracy)

#             avg_accuracy = sum(Accuracy) / len(Accuracy)

#             if avg_accuracy > best_accuracy:
#                 best_accuracy = avg_accuracy
#                 best_params = (c1, k1, Ie)

# # Output results
# print(f"\nBest Accuracy: {best_accuracy:.2f}%")
# print(f"Best Parameters:\nc1 = {best_params[0]:.4f}, k1 = {best_params[1]:.4f}, Ie = {best_params[2]:.4f}")

# -----------------------------------------------------------------------------------------------------

# import numpy as np
# from scipy.optimize import minimize
# from check_data import dat_array as load_data

# # Data runs to evaluate over
# run_ids = (4, 5, 6, 7, 12, 13)

# def simulate_and_score(c1, k1, Ie):
#     if Ie <= 0:
#         return 1e6  # penalize invalid values
    
#     accuracy_list = []
#     kg = -0.22
#     dt = 0.001
#     n_steps = 7001

#     for run_nr in run_ids:
#         DeltaElev = load_data(f"run{run_nr}/aircraft/DeltElev")
#         IservoElev = load_data(f"run{run_nr}/aircraft/IservoElev")

#         A = np.array([[-(c1 / Ie), -(k1 / Ie)], [1, 0]])
#         B = np.array([[-kg / Ie], [0]])

#         x = np.zeros((2, 1))
#         xlist = np.zeros(n_steps)

#         for i in range(n_steps):
#             u = np.array([[IservoElev[i]]])
#             xdot = A @ x + B @ u
#             x += xdot * dt
#             xlist[i] = x[1, 0]  # store deflection output

#         absolute_error = np.abs(xlist - DeltaElev)
#         error_norm = np.linalg.norm(absolute_error) / np.linalg.norm(DeltaElev)
#         accuracy = (1 - error_norm) * 100
#         accuracy_list.append(accuracy)

#     avg_accuracy = sum(accuracy_list) / len(accuracy_list)
#     return -avg_accuracy  # minimize negative accuracy = maximize accuracy

# # Wrap in function for optimizer
# def objective(params):
#     c1, k1, Ie = params
#     return simulate_and_score(c1, k1, Ie)

# # Initial guess and bounds (based on your current best)
# initial_guess = [1.08, 9.16, 0.03]
# bounds = [
#     (1.07, 1.09),     # narrower around c1
#     (9.12, 9.20),     # narrower around k1
#     (0.025, 0.05),    # expand upper Ie range
# ]

# # Run optimization
# result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')

# # Output best result
# best_c1, best_k1, best_Ie = result.x
# print(f"\nOptimized Accuracy: {-result.fun:.2f}%")
# print(f"Optimized Parameters:")
# print(f"  c1 = {best_c1:.4f}")
# print(f"  k1 = {best_k1:.4f}")
# print(f"  Ie = {best_Ie:.4f}")


# ----------------------------------------------------------------------------------

import numpy as np
from scipy.optimize import differential_evolution
from check_data import dat_array as load_data
import matplotlib.pyplot as plt

# Data runs to evaluate over
run_ids = (4, 5, 6, 7, 12, 13)

# Function to simulate the model and return negative accuracy (for minimization)
def simulate_and_score(c1, k1, Ie):
    if Ie <= 0:
        return 1e6  # Penalize invalid values
    
    accuracy_list = []
    kg = -0.22
    dt = 0.001
    n_steps = 7001

    for run_nr in run_ids:
        DeltaElev = load_data(f"run{run_nr}/aircraft/DeltElev")
        IservoElev = load_data(f"run{run_nr}/aircraft/IservoElev")

        # System matrices
        A = np.array([[-(c1 / Ie), -(k1 / Ie)], [1, 0]])
        B = np.array([[-kg / Ie], [0]])

        x = np.zeros((2, 1))  # Initial state [angle, velocity]
        xlist = np.zeros(n_steps)  # List to store simulation result

        # Time integration loop (Euler method)
        for i in range(n_steps):
            u = np.array([[IservoElev[i]]])  # Control input
            xdot = A @ x + B @ u
            x += xdot * dt
            xlist[i] = x[1, 0]  # Store control surface deflection

        # Compute absolute error
        absolute_error = np.abs(xlist - DeltaElev)
        error_norm = np.linalg.norm(absolute_error) / np.linalg.norm(DeltaElev)
        accuracy = (1 - error_norm) * 100  # Convert to percentage accuracy
        accuracy_list.append(accuracy)

    # Return negative average accuracy for optimization (since we minimize)
    avg_accuracy = sum(accuracy_list) / len(accuracy_list)
    return -avg_accuracy  # We want to maximize accuracy, so minimize the negative value

# Objective function for differential evolution (uses the simulate_and_score function)
def objective(params):
    c1, k1, Ie = params
    return simulate_and_score(c1, k1, Ie)

# Set bounds for each parameter (c1, k1, Ie) based on the previous results
bounds = [
    (1.05, 1.15),  # c1 range (tightened from earlier optimization)
    (9.0, 9.25),   # k1 range (tightened from earlier optimization)
    (0.025, 0.05)  # Ie range (expanded slightly from earlier optimization)
]

# Run the differential evolution optimization
result = differential_evolution(objective, bounds, strategy='best1bin', maxiter=20, disp=True)

# Extract the optimized parameters and accuracy
best_c1, best_k1, best_Ie = result.x
optimized_accuracy = -result.fun  # Convert back to positive accuracy

# Output the results
print(f"\nOptimized Accuracy: {optimized_accuracy:.2f}%")
print(f"Optimized Parameters:")
print(f"  c1 = {best_c1:.4f}")
print(f"  k1 = {best_k1:.4f}")
print(f"  Ie = {best_Ie:.4f}")

# Optional: Visualize the result using one of the runs (e.g., run 4)
run_nr = 4  # You can choose another run if you wish to check

# Load actual data
DeltaElev = load_data(f"run{run_nr}/aircraft/DeltElev")
IservoElev = load_data(f"run{run_nr}/aircraft/IservoElev")

# System matrices with the best parameters
kg = -0.22
A = np.array([[-(best_c1 / best_Ie), -(best_k1 / best_Ie)], [1, 0]])
B = np.array([[-kg / best_Ie], [0]])

# Initial state and simulation loop
x = np.zeros((2, 1))
xlist = np.zeros(7001)

# Time integration loop (Euler method)
dt = 0.001
for i in range(7001):
    u = np.array([[IservoElev[i]]])
    xdot = A @ x + B @ u
    x += xdot * dt
    xlist[i] = x[1, 0]  # Store control surface deflection

# Plot comparison of simulated vs actual Delta Elevation
time_steps = np.linspace(0, 7000, 7001)
plt.figure(figsize=(10, 6))
plt.plot(time_steps, DeltaElev, label='Actual DeltaElev', color='blue')
plt.plot(time_steps, xlist, label='Simulated DeltaElev', color='red', linestyle='dashed')
plt.xlabel('Time Steps')
plt.ylabel('Delta Elevation')
plt.title(f'Optimized Model vs Actual (Run {run_nr})')
plt.legend()
plt.show()
