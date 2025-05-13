from scipy.optimize import differential_evolution

def objective(params):
    k1, k2, c2 = params
    # Run your model with these parameters (c1 stays constant)
    _, run1_acc2 = model2(4, array, resolution, flip, divfactor, k_g, k1, k2, c1_numvalue, c2, a_velo, extragraphs, showmainplots, printeigenvalues)
    _, run3_acc2 = model2(5, array, resolution, flip, divfactor, k_g, k1, k2, c1_numvalue, c2, a_velo, extragraphs, showmainplots, printeigenvalues)
    # ...repeat for all runs...
    accuracy_DOF2 = [run1_acc2, run3_acc2, ...]
    return -np.mean(accuracy_DOF2)  # Negative for maximization

bounds = [(100000, 1000000), (10, 40), (1, 20)]  # Example bounds for k1, k2, c2
result = differential_evolution(objective, bounds)
print("Best parameters:", result.x)
print("Best accuracy:", -result.fun)