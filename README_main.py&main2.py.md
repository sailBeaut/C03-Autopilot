# Model 1 Aileron & Elevator Files
This Python script runs the model and allows the user to choose between plotting the deflection over time and printing the accuracies. All parameters are also defined in this file.
## File Overview
main.py:                            Run aileron model
main2.py:                           Run elevator model
filtered_data_nn_2.hdf5             Data file processed to cut out unsuitable data
check_data.py:                      Reads data from processed data file


## How to Use
1. **Place Required Files Together**  
   Ensure the following files are in the same directory:
   - `main.py` (aileron model)
   - `main.py` (elevator model)
   - `filtered_data_nn_2.hdf5` (processed data file)
   - `check_data.py` (functions to read data)

2. **Adjust Parameters (Optional)**  
   In the main loop, the following model parameters are defined, which can be altered to change the model:
   - Stiffness (`k1`)
   - Damping (`c1`)
   - Moment of inertia (`Ie`)

3. **Adjust configuration**
   On lines 7 and 8 you can the function of the program by changing values to True or False
   - `plot_all`            - this turns the plots of displacement against time on or off
   - `print_accuracies`    - this prints the accuracy per run and the average accuracy over all the runs

4. **Run the Script**
Running main1 for the aileron, or main2 for the elevator will:
- Run the model across the selected test cases.
- Calculate the accuracies of the selected runs, if enabled
- Plot the results compared to measured results, if enabled

## Notes
- The parameters that are currently filled in for k1, c1 and Ie, are the values that were found to be optimal.
- No iterative optimization algorithm was used to find these values, instead this was done through trial and error.
