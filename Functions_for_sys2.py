
import sympy as sp
from check_fulldata import dat_array, dat_array_ground
import numpy as np
import matplotlib.pyplot as plt
from Testkernels import smooth_data


def model2(run, array, resolution, flip, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues):
    # Load data
    Delta = []
    DeltaDrum = []
    Iservo = []
    Dynpress = []

    if run == 0:
        # Load data for ground run
        Delta = dat_array_ground("data/aircraft/data/DeltaAil")
        DeltaDrum = dat_array_ground("data/aircraft/data/DeltaDrumAil")
        Iservo = dat_array_ground("data/aircraft/data/IservoAil")
        Dynpress = dat_array_ground("data/aircraft/data/DynPress")

    elif run in (1, 3, 8, 9, 10, 11):
        Delta = dat_array(f"run{run}/aircraft/DeltaAil")
        DeltaDrum = dat_array(f"run{run}/aircraft/DeltaDrumAil")
        Iservo = dat_array(f"run{run}/aircraft/IservoAil")
        Dynpress = dat_array(f"run{run}/aircraft/DynPress")

    elif run in (4, 5, 6, 7, 12, 13):
        Delta = dat_array(f"run{run}/aircraft/DeltaElev")
        DeltaDrum = dat_array(f"run{run}/aircraft/DeltaDrumElev")
        Iservo = dat_array(f"run{run}/aircraft/IservoElev")
        Dynpress = dat_array(f"run{run}/aircraft/DynPress")
    
    Delta = Delta[:-10]
    DeltaDrum = DeltaDrum[:-10]
    Iservo = Iservo[:-10]
    Dynpress = Dynpress[:-10]

    # Step 1: Define symbolic variables for Mass (M), Damping (C), and Stiffness (K)
    j1, j2 = sp.symbols('j1 j2')  # Masses
    k1, k2 = sp.symbols('k1 k2')  # Stiffness values
    c1, c2 = sp.symbols('c1 c2')  # Damping coefficients
    t = sp.symbols('t')
    r1 = sp.symbols('r1')
    r2 = sp.symbols('r2')
    l = sp.symbols('l')

    # Define mass, stiffness, and damping matrices
    M = sp.Matrix([[j1, 0], [0, j2]])  # Mass matrix
    C = sp.Matrix([[c1 * r1 ** 2, -c1 * r1 * r2], [-c1 * r1 * r2, c2 + c1 * r2 ** 2]])  # Damping matrix
    K = sp.Matrix([[k1 * r1 ** 2, -k1 * r1 * r2], [-k1 * r1 * r2, k2 + k1 * r2 ** 2]])  # Stiffness matrix

    # Define force vector (external forces)
    F = sp.Matrix([sp.sin(t), 0])  # Example force applied to first DOF
    # Assume input Torque is sinus function

    # Compute M inverse
    M_inv = M.inv()

    # Step 2: Define first-order system
    x = sp.Matrix(sp.symbols('x1 x2'))  # Displacement vector
    v = sp.Matrix(sp.symbols('v1 v2'))  # Velocity vector

    # Combine into state-space form
    Y = sp.Matrix.vstack(x, v)

    # Convert symbolic to numerical
    subs_dict = {j1: 5.4E-5, j2: 7.97E-2, k1: k1_numvalue, k2: k2_numvalue, c1: c1_numvalue, c2: c2_numvalue, r1: 2.52E-2, r2: 7.9E-2}
    M_num = np.array(M.subs(subs_dict)).astype(np.float64)
    K_num = np.array(K.subs(subs_dict)).astype(np.float64)
    C_num = np.array(C.subs(subs_dict)).astype(np.float64)

    # Calculate eigenvalues
    j1_value, j2_value, k1_value, k2_value, c1_value, c2_value, r1_value, r2_value = subs_dict[j1], subs_dict[j2], \
    subs_dict[k1], subs_dict[k2], subs_dict[c1], subs_dict[c2], subs_dict[r1], subs_dict[r2]

    def eigenvalues(j1_value, j2_value, k1_value, k2_value, c1_value, c2_value, r1_value, r2_value):
        j1, j2, k1, k2, c1, c2, r1, r2 = j1_value, j2_value, k1_value, k2_value, c1_value, c2_value, r1_value, r2_value
        A = np.array(
            [[0, 0, 1, 0], [0, 0, 0, 1], [-k1 * r1 ** 2 / j1, k1 * r1 * r2 / j1, -c1 * r1 ** 2 / j1, c1 * r1 * r2 / j1],
            [k1 * r1 * r2 / j2, -(k2 + k1 * r2 ** 2) / j2, c1 * r1 * r2 / j2, -(c2 + c1 * r2 ** 2) / j2]])

        # Calculate the eigenvalues
        eigenvalues = np.linalg.eigvals(A)

        # Print the eigenvalues
        print("Eigenvalues of the matrix are:", eigenvalues)


    # Convert system equations to NumPy function
    def system(Y, t):
        x = Y[:2]  # First two elements are displacements
        v = Y[2:]  # Last two elements are velocities
        F_num = np.array([np.interp(t, t_values, Iservo) * k_g, -np.interp(t, t_values, Dynpress)* a_velo])  # Numerical force

        dxdt = v
        dvdt = np.linalg.inv(M_num) @ (F_num - C_num @ v - K_num @ x)

        return np.hstack((dxdt, dvdt))

    
    # Step 3: Implement RK4 for numerical integration
    def runge_kutta4(f, Y0, t):
        n = len(t)
        h = t[1] - t[0]  # Time step
        Y = np.zeros((n, len(Y0)))  # Store results
        Y[0] = Y0

        for i in range(n - 1):
            k1 = f(Y[i], t[i]) # First derivative at the current time step
            k2 = f(Y[i] + 0.5 * h * k1, t[i] + 0.5 * h) # Second derivative at the midpoint
            k3 = f(Y[i] + 0.5 * h * k2, t[i] + 0.5 * h) # Third derivative at the midpoint
            k4 = f(Y[i] + h * k3, t[i] + h) # Fourth derivative at the next time step

            Y[i + 1] = Y[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            if(abs(Y[i + 1][1] - Y[i][1]) <= 0.00001):
                Y[i + 1][1] = Y[i][1]
            #if(abs(Y[i + 1][1] - Y[i][1]) >= 0.052):
            #    Y[i + 1][1] = Y[i][1]
        return Y


    # Time settings

    t_values_high_res = np.linspace(0, (len(Delta) - 1), len(Delta) * resolution) / 1000  # High-resolution time array
    t_values = np.linspace(0, len(Delta) - 1, len(Delta)  ) / 1000  # Original time array for plotting

    # Time step (assuming uniform time steps in t_values)
    dt = t_values[1] - t_values[0]

# Initial conditions using 3-point forward difference for velocity
    #Smooth data for initial conditions
    DeltaDrum_smooth = smooth_data(DeltaDrum, 5)
    Delta_smooth = smooth_data(Delta, 5)

    Y0 = [
    DeltaDrum[0]*divfactor,  # Initial displacement for DOF1
    flip * Delta[0],  # Initial displacement for DOF2
    (-3 * DeltaDrum_smooth[0] + 4 *DeltaDrum_smooth[1] - DeltaDrum_smooth[2]) / (2 * dt),  # Initial velocity for DOF1
    flip * (-3 * Delta_smooth[0] + 4 * Delta_smooth[1] - Delta_smooth[2]) / (2 * dt)  # Initial velocity for DOF2
    ]

    # Solve using RK4 with high-resolution time steps
    Y_sol_high_res = runge_kutta4(system, Y0, t_values_high_res)

    # Interpolate the high-resolution solution back to the original time steps
    Y_sol = np.zeros((len(t_values), Y_sol_high_res.shape[1]))
    for i in range(Y_sol_high_res.shape[1]):
        Y_sol[:, i] = np.interp(t_values, t_values_high_res, Y_sol_high_res[:, i])
        

    if printeigenvalues == True:
        eigenvalues(j1_value, j2_value, k1_value, k2_value, c1_value, c2_value, r1_value, r2_value)

    #Step 5: Calc Accuracy
    absolute_error1 = np.abs(Y_sol[:, 0]/divfactor - DeltaDrum)
    absolute_error2 = np.abs(flip * Y_sol[:, 1] - Delta)

    # Compute accuracy as percentage
    error_norm1 = np.linalg.norm(absolute_error1) / np.linalg.norm(DeltaDrum)
    accuracy1 = (1 - error_norm1) * 100
    error_norm2 = np.linalg.norm(absolute_error2) / np.linalg.norm(Delta)
    accuracy2 = (1 - error_norm2) * 100

    # Print accuracy
    print(f"Model Accuracy of DOF1 of run{run}: {accuracy1:.2f}%")
    print(f"Model Accuracy of DOF2 of run{run}: {accuracy2:.2f}%")

    #Check Array
    if array == True:
        print("DOF1:", Y_sol[:, 0])
        print("DeltaDrum:", DeltaDrum)
        print("DeltaDrumSmooth:", DeltaDrum_smooth)
        print("DOF2:", Y_sol[:, 1])
        print("Delta:", Delta)
        print("DeltaSmooth:", Delta_smooth)

    if showmainplots == True:
        # Step 6: Plot results
        #Plot of DOF2 compared to DeltaAil
        plt.subplot(2, 4, 1)
        plt.plot(t_values, flip * Y_sol[:, 1], label="x2 (DOF 2)", color="blue")
        plt.plot(t_values, Delta, label="Delta", color="orange")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement of DOF 2")
        plt.title(f"Model Accuracy of DOF2: {accuracy2:.2f}%")
        plt.legend()
        plt.grid()

        #Plot of DOF2' Accuracy compared to DeltaAil
        plt.subplot(2, 4, 2)
        percentage_error_dof2 = np.abs((flip * Y_sol[:, 1] - Delta) / Delta) * 100
        plt.plot(t_values, percentage_error_dof2, label="x2 (DOF 2)", color = "red")
        plt.xlabel("Time (s)")
        plt.ylabel("Percentage Error of DOF 2 (%)")
        plt.title("Error")
        plt.legend()
        plt.grid()

        #Plot of DOF2 separate
        plt.subplot(2, 4, 3)
        plt.plot(t_values, flip * Y_sol[:, 1], label="x2 (DOF 2)", color="blue")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement of DOF 2")
        plt.title(f"run{run}")
        plt.legend()
        plt.grid()

        #Plot of DeltaAil separate
        plt.subplot(2, 4, 4)
        plt.plot(t_values, Delta, label="Delta", color="orange")
        plt.plot(t_values, Delta_smooth, label="DeltaSmooth", color="green")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement of DOF 2")
        plt.legend()
        plt.grid()

        #Plot of DOF1 compared to DeltaDrumAil
        plt.subplot(2, 4, 5)
        plt.plot(t_values, Y_sol[:, 0]/divfactor, label="x1 (DOF 1)", color="blue")
        plt.plot(t_values, DeltaDrum, label="DeltaDrum", color="orange")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement of DOF 1")
        plt.title(f"Model Accuracy of DOF1: {accuracy1:.2f}%")
        plt.legend()
        plt.grid()

        #Plot of DOF1' Accuracy compared to DeltaDrumAil
        plt.subplot(2, 4, 6)
        percentage_error_dof1 = np.abs((Y_sol[:, 0]/divfactor - DeltaDrum) / DeltaDrum) * 100
        plt.plot(t_values, percentage_error_dof1, label="x1 (DOF 1)", color = "red")
        plt.xlabel("Time (s)")
        plt.ylabel("Percentage Error of DOF 1 (%)")
        plt.title("Error")
        plt.legend()
        plt.grid()

        #Plot of DOF1 separate
        plt.subplot(2, 4, 7)
        plt.plot(t_values, Y_sol[:, 0]/divfactor, label="x1 (DOF 1)", color="blue")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement of DOF 1")
        plt.legend()
        plt.grid()

        #Plot of DeltaDrumAil separate
        plt.subplot(2, 4, 8)
        plt.plot(t_values, DeltaDrum, label="DeltaDrum", color="orange")
        plt.plot(t_values, DeltaDrum_smooth, label="DeltaDrumSmooth", color="green")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement of DOF 1")
        plt.legend()
        plt.grid()
        plt.show()

    if extragraphs == True:
        #Plot of DOF1 and DOF2 in one graph
        plt.plot(t_values, Y_sol[:, 0]*(180*np.pi)/divfactor, label="x1 (DOF 1)")
        plt.plot(t_values, flip * Y_sol[:, 1]*(180*np.pi), label="x2 (DOF 2)")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement")
        plt.legend()
        plt.grid()
        plt.show()

        #Plot of the cable slack
        plt.plot(t_values, (flip * Y_sol[:, 1]*(180*np.pi))/(Y_sol[:, 0]*(180*np.pi)/divfactor), label="Linearity vs. cable slack")
        plt.xlabel("Time (s)")
        plt.ylabel("Ratio")
        plt.title("Linearity")
        plt.legend()
        plt.grid()
        plt.show()
    return accuracy1, accuracy2


def accuracy_plot_ail(accuracy_dof1_array, accuracy_dof2_array):
    #Plot of accuracy of DOF1 and DOF2 between runs 1,3,8,9,10,11
    avg_acc1 = sum(accuracy_dof1_array) / len(accuracy_dof1_array)
    avg_acc2 = sum(accuracy_dof2_array) / len(accuracy_dof2_array)
    avg_acc1_wo9n11 = (sum(accuracy_dof1_array[:-1])-accuracy_dof1_array[3]) / len(accuracy_dof1_array[:-2])
    avg_acc2_wo9n11 = (sum(accuracy_dof2_array[:-1])-accuracy_dof2_array[3]) / len(accuracy_dof2_array[:-2])
    avg_acc1_wo11 = sum(accuracy_dof1_array[:-1]) / len(accuracy_dof1_array[:-1])
    avg_acc2_wo11 = sum(accuracy_dof2_array[:-1]) / len(accuracy_dof2_array[:-1])

    plt.plot(accuracy_dof1_array, label="DOF1", color="red")
    plt.plot(accuracy_dof2_array, label="DOF2", color="blue")
    plt.axhline(y=avg_acc1, color='r', linestyle='--', label=f"Average DOF1: {avg_acc1:.2f}%")
    plt.axhline(y=avg_acc2, color='b', linestyle='--', label=f"Average DOF2: {avg_acc2:.2f}%")
    plt.axhline(y=avg_acc1_wo11, color='green', linestyle=':', label=f"Average DOF1 (no run11): {avg_acc1_wo11:.2f}%")
    plt.axhline(y=avg_acc2_wo11, color='orange', linestyle=':', label=f"Average DOF2 (no run11): {avg_acc2_wo11:.2f}%")
    plt.axhline(y=avg_acc1_wo9n11, color='purple', linestyle='-.', label=f"Average DOF1 (no run9 and run11): {avg_acc1_wo9n11:.2f}%")
    plt.axhline(y=avg_acc2_wo9n11, color='yellow', linestyle='-.', label=f"Average DOF2 (no run9 and run11): {avg_acc2_wo9n11:.2f}%")
    plt.xlabel("Run")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Aileron(Run 1,3,8,9,10,11)")
    plt.legend()
    plt.grid()
    plt.show()

def accuracy_plot_elev(accuracy_dof1_array, accuracy_dof2_array):
    #Plot of accuracy of DOF1 and DOF2 between runs 4,5,6,7,12,13
    avg_acc1 = sum(accuracy_dof1_array) / len(accuracy_dof1_array)
    avg_acc2 = sum(accuracy_dof2_array) / len(accuracy_dof2_array)

    plt.plot(accuracy_dof1_array, label="DOF1", color="red")
    plt.plot(accuracy_dof2_array, label="DOF2", color="blue")
    plt.axhline(y=avg_acc1, color='r', linestyle='--', label=f"Average DOF1: {avg_acc1:.2f}%")
    plt.axhline(y=avg_acc2, color='b', linestyle='--', label=f"Average DOF2: {avg_acc2:.2f}%")
    plt.xlabel("Run")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Elevator(Run 4,5,6,7,12,13)")
    plt.legend()
    plt.grid()
    plt.show()