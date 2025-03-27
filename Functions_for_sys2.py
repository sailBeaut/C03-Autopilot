
import sympy as sp
from check_fulldata import dat_array
import numpy as np
import matplotlib.pyplot as plt
from Testkernels import current_smoothed_ma


def model2_aileron(run, divfactor, k_g, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue, a_velo, extragraphs, showmainplots, printeigenvalues):
    # Load data
    DeltaAil = dat_array(f"run{run}/aircraft/DeltaAil")
    DeltaDrumAil = dat_array(f"run{run}/aircraft/DeltaDrumAil")
    IservoAil = dat_array(f"run{run}/aircraft/IservoAil")
    Dynpress = dat_array(f"run{run}/aircraft/DynPress")

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
        F_num = np.array([np.interp(t, t_values, IservoAil) * k_g, -np.interp(t, t_values, Dynpress)* a_velo])  # Numerical force

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
            k1 = f(Y[i], t[i])
            k2 = f(Y[i] + 0.5 * h * k1, t[i] + 0.5 * h)
            k3 = f(Y[i] + 0.5 * h * k2, t[i] + 0.5 * h)
            k4 = f(Y[i] + h * k3, t[i] + h)

            Y[i + 1] = Y[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return Y


    # Time settings
    t_values = np.linspace(0, len(DeltaAil) - 1, len(DeltaAil)) / 1000  # Time in seconds, with length matching DeltaAil
    Y0 = [DeltaDrumAil[0], -DeltaAil[0], ((DeltaDrumAil[0]-DeltaDrumAil[1])/0.001), -(DeltaAil[0]-DeltaAil[1])/0.001]  # Initial conditions: x1 = x2 = v1 = v2 = 0

    # Solve using RK4
    Y_sol = runge_kutta4(system, Y0, t_values)

    if printeigenvalues == True:
        eigenvalues(j1_value, j2_value, k1_value, k2_value, c1_value, c2_value, r1_value, r2_value)

    #Step 5: Calc Accuracy
    absolute_error1 = np.abs(Y_sol[:, 0]/divfactor - DeltaDrumAil)
    absolute_error2 = np.abs(-Y_sol[:, 1] - DeltaAil)

    # Compute accuracy as percentage
    error_norm1 = np.linalg.norm(absolute_error1) / np.linalg.norm(DeltaDrumAil)
    accuracy1 = (1 - error_norm1) * 100
    error_norm2 = np.linalg.norm(absolute_error2) / np.linalg.norm(DeltaAil)
    accuracy2 = (1 - error_norm2) * 100

    # Print accuracy
    print(f"Model Accuracy of DOF1 of run{run}: {accuracy1:.2f}%")
    print(f"Model Accuracy of DOF2 of run{run}: {accuracy2:.2f}%")


    if showmainplots == True:
        # Step 6: Plot results
        plt.subplot(2, 4, 1)
        plt.plot(t_values, -Y_sol[:, 1], label="x2 (DOF 2)")
        plt.plot(t_values, DeltaAil, label="DeltaAil")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement of DOF 2")
        plt.title(f"Model Accuracy of DOF2: {accuracy2:.2f}%")
        plt.legend()
        plt.grid()

        plt.subplot(2, 4, 2)
        percentage_error_dof2 = np.abs((-Y_sol[:, 1] - DeltaAil) / DeltaAil) * 100
        plt.plot(t_values, percentage_error_dof2, label="x2 (DOF 2)")
        plt.xlabel("Time (s)")
        plt.ylabel("Percentage Error of DOF 2 (%)")
        plt.title("Error")
        plt.legend()
        plt.grid()


        plt.subplot(2, 4, 3)
        plt.plot(t_values, -Y_sol[:, 1], label="x2 (DOF 2)")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement of DOF 2")
        plt.title(f"run{run}")
        plt.legend()
        plt.grid()

        plt.subplot(2, 4, 4)
        plt.plot(t_values, DeltaAil, label="DeltaAil")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement of DOF 2")
        plt.title("MDOF System Response (RK4)")
        plt.legend()
        plt.grid()

        plt.subplot(2, 4, 5)
        plt.plot(t_values, Y_sol[:, 0]/divfactor, label="x1 (DOF 1)")
        plt.plot(t_values, DeltaDrumAil, label="DeltaDrumAil")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement of DOF 1")
        plt.title(f"Model Accuracy of DOF1: {accuracy1:.2f}%")
        plt.legend()
        plt.grid()

        plt.subplot(2, 4, 6)
        percentage_error_dof1 = np.abs((Y_sol[:, 0]/divfactor - DeltaDrumAil) / DeltaDrumAil) * 100
        plt.plot(t_values, percentage_error_dof1, label="x1 (DOF 1)")
        plt.xlabel("Time (s)")
        plt.ylabel("Percentage Error of DOF 1 (%)")
        plt.title("Error")
        plt.legend()
        plt.grid()

        plt.subplot(2, 4, 7)
        plt.plot(t_values, Y_sol[:, 0]/divfactor, label="x1 (DOF 1)")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement of DOF 1")
        plt.title("MDOF System Response (RK4)")
        plt.legend()
        plt.grid()


        plt.subplot(2, 4, 8)
        plt.plot(t_values, DeltaDrumAil, label="DeltaDrumAil")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement of DOF 1")
        plt.title("MDOF System Response (RK4)")
        plt.legend()
        plt.grid()
        plt.show()

    if extragraphs == True:
        plt.plot(t_values, Y_sol[:, 0]*(180*np.pi)/divfactor, label="x1 (DOF 1)")
        plt.plot(t_values, -Y_sol[:, 1]*(180*np.pi), label="x2 (DOF 2)")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement")
        plt.title("MDOF System Response (RK4)")
        plt.legend()
        plt.grid()
        plt.show()


        plt.plot(t_values, (-Y_sol[:, 1]*(180*np.pi))/(Y_sol[:, 0]*(180*np.pi)/divfactor), label="Linearity vs. cable slack")
        plt.xlabel("Time (s)")
        plt.ylabel("Ratio")
        plt.title("Linearity")
        plt.legend()
        plt.grid()
        plt.show()
    return accuracy1, accuracy2


def accuracy_plot(accuracy_dof1_array, accuracy_dof2_array):
    avg_acc1 = sum(accuracy_dof1_array) / len(accuracy_dof1_array)
    avg_acc2 = sum(accuracy_dof2_array) / len(accuracy_dof2_array)

    plt.plot(accuracy_dof1_array, label="DOF1", color="red")
    plt.plot(accuracy_dof2_array, label="DOF2", color="blue")
    plt.axhline(y=avg_acc1, color='r', linestyle='--', label=f"Average DOF1: {avg_acc1:.2f}%")
    plt.axhline(y=avg_acc2, color='b', linestyle='--', label=f"Average DOF2: {avg_acc2:.2f}%")
    plt.xlabel("Run")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Aileron (Run 1,3,8,9,10,11)")
    plt.legend()
    plt.grid()
    plt.show()