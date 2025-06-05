# Model2 Optimization Script
This Python script performs an optimization process for a physical simulation model (`model2`) to maximize the accuracy of Degree of Freedom 2 (DOF 2). It iteratively adjusts model parameters and evaluates the accuracy across several predefined test runs. There are two possible usages/ways to use this script; the first one is set the accuracy you want the program to reach and let it run until it finishes. This is not recommended when  the accuracy is too high because it simply won't reach it. The suggested use of this program for the later stages of optimisation is, by iteratively adjusting one parameter at a time, and let it run. You will have to pay attention and seek for the local maximum by inspection. Once this is reached, go on to the next parameter and repeat the process. 

## File Overview
tuning parameter.py:          This script
Functions_for_sys2.py:        Contains the model2 and accuracy_plot_elev functions
check_data.py:                Provides data structures and print utilities


## How to Use
1. **Place Required Files Together**  
   Ensure the following files are in the same directory:
   - `tuning parameter.py` (the script in question)
   - `Functions_for_sys2.py` (providing `model2` and `accuracy_plot_elev`)
   - `check_data.py` (containing `dat_array` and `print_struc`)

2. **Adjust Parameters (Optional)**  
   At the top of the script, you can configure the initial parameters:
   - Stiffness (`k1_numvalue`, `k2_numvalue`)
   - Damping (`c1_numvalue`, `c2_numvalue`)
   - Velocity factor (`a_velo`)
   - Flattening coefficient (`flatten_coeff`)
   - Etcetera

3. **Run the Script**
The script will:
- Run the `model2` simulation across 6 test cases.
- Adjust parameters to maximise the average accuracy of DOF 2.
- Print intermediate and final best accuracy and parameters.
- Plot the final accuracy results for DOF 2 using `accuracy_plot_elev`.

## Testing trial
To make sure the code functioning is understood parameters can be tested.  

## Notes
- **Stopping Condition**: The optimisation loop stops once the average accuracy of DOF 2 exceeds 85%, or another accuracy you want it to achieve.
- **Plotting**: The final plot displays the DOF 2 accuracy for each test case.
- **Customization**: You can uncomment and tweak the parameter update lines to experiment with different optimization strategies, as explained in the description of the script.

