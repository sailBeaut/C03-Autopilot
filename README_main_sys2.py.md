# Model 2 Main File
This Python script runs the model and allows the user to choose between configurations. Also the parameters are defined in this file
## File Overview
Main System 2:                      Run the model and make altercations
Functions_for_sys2.py:              Contains the model2 and accuracy functions
check_fulldata.py:                  Provides data structures and print utilities


## How to Use
1. **Place Required Files Together**  
   Ensure the following files are in the same directory:
   - `main_sys2.py` (the script in question)
   - `Functions_for_sys2.py` (providing `model2`, `accuracy_plot_elev` and `accuracy_plot_ail`)
   - `check_data.py` (containing `dat_array` and `print_struc`)

2. **Adjust Parameters (Optional)**  
   At the top of the script, you can configure the initial parameters:
   - Stiffness (`k1_numvalue`, `k2_numvalue`)
   - Damping (`c1_numvalue`, `c2_numvalue`)
   - Velocity factor (`a_velo`)
   - Flattening coefficient (`flatten_coeff`)
   - Etcetera

3. **Adjust configuration**
   After the comment 'On or Off', you can make configuration on or off by change to true or false:
   - `ground`              - this turns the ground test runs on or off
   - `aileron`             - this turns the aileron test runs on or off
   - `elevator`            - this turns the elevator test runs on or off
   - `array`               - this turns the print statement of the model array on or off
   - `extragraphs`         - this turns the plots about linearity on or off
   - `showmainplots`       - this turns the main plots on or off, without this on only the accuracy is printed
   - `printeigenvalues`    - this turns the print statement of the eigenvalues of the system on or off

4. **Run the Script**
The script will:
- Run the `model2` simulation across 6 test cases.
- Adjust parameters to maximise the average accuracy of DOF 2.
- Print intermediate and final best accuracy and parameters.
- Plot the final accuracy results for DOF 2 using `accuracy_plot_elev`.

## Notes
- **Stopping Condition**: The optimisation loop stops once the average accuracy of DOF 2 exceeds 85%, or another accuracy you want it to achieve.
- **Plotting**: The final plot displays the DOF 2 accuracy for each test case.
- **Customization**: You can uncomment and tweak the parameter update lines to experiment with different optimization strategies, as explained in the description of the script.
