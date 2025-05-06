import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from check_fulldata import dat_array, dat_array_ground


class MatrixModelNN(nn.Module):
    def __init__(self, M_num, C_num, K_num):
        super(MatrixModelNN, self).__init__()
        self.M_num = torch.tensor(M_num, dtype=torch.float32)
        self.C_num = torch.tensor(C_num, dtype=torch.float32)
        self.K_num = torch.tensor(K_num, dtype=torch.float32)

    def forward(self, Dynpress, Iservo):
        # Convert inputs to tensors
        Dynpress = torch.tensor(Dynpress, dtype=torch.float32).unsqueeze(1)  # Shape: (batch_size, 1)
        Iservo = torch.tensor(Iservo, dtype=torch.float32).unsqueeze(1)      # Shape: (batch_size, 1)

        # Combine inputs into force vector
        F_num = torch.cat([Iservo, -Dynpress], dim=1)  # Shape: (batch_size, 2)

        # Solve for accelerations: dv/dt = M^-1 * (F - C*v - K*x)
        # For simplicity, assume v and x are zero initially
        dvdt = torch.linalg.solve(self.M_num, F_num.T).T  # Shape: (batch_size, 2)

        return dvdt  # Output accelerations


def train_nn(Dynpress, Iservo, M_num, C_num, K_num, epochs=100, lr=0.01):
    # Initialize the model
    model = MatrixModelNN(M_num, C_num, K_num)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Generate dummy target data (replace with actual data)
    target = np.zeros((len(Dynpress), 2))  # Replace with actual target accelerations

    # Convert data to tensors
    Dynpress_tensor = torch.tensor(Dynpress, dtype=torch.float32)
    Iservo_tensor = torch.tensor(Iservo, dtype=torch.float32)
    target_tensor = torch.tensor(target, dtype=torch.float32)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(Dynpress_tensor, Iservo_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model


def plot_results(t_values, Delta, DeltaDrum, Y_sol):
    # Plot DOF2 compared to DeltaAil
    plt.subplot(2, 4, 1)
    plt.plot(t_values, Y_sol[:, 1], label="x2 (DOF 2)", color="blue", marker='o')
    plt.plot(t_values, Delta, label="Delta", color="orange", marker='o')
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement of DOF 2")
    plt.title("DOF2 vs Delta")
    plt.legend()
    plt.grid()

    # Plot DOF1 compared to DeltaDrumAil
    plt.subplot(2, 4, 5)
    plt.plot(t_values, Y_sol[:, 0], label="x1 (DOF 1)", color="blue", marker='o')
    plt.plot(t_values, DeltaDrum, label="DeltaDrum", color="orange", marker='o')
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement of DOF 1")
    plt.title("DOF1 vs DeltaDrum")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


# Example usage
def model2_nn(run, k1_numvalue, k2_numvalue, c1_numvalue, c2_numvalue):
    # Load data
    if run == 0:
        Delta = dat_array_ground("data/aircraft/data/DeltaAil")
        DeltaDrum = dat_array_ground("data/aircraft/data/DeltaDrumAil")
        Iservo = dat_array_ground("data/aircraft/data/IservoAil")
        Dynpress = dat_array_ground("data/aircraft/data/DynPress")
    else:
        Delta = dat_array(f"run{run}/aircraft/DeltaAil")
        DeltaDrum = dat_array(f"run{run}/aircraft/DeltaDrumAil")
        Iservo = dat_array(f"run{run}/aircraft/IservoAil")
        Dynpress = dat_array(f"run{run}/aircraft/DynPress")

    # Define matrices
    j1, j2 = 5.4E-5, 7.97E-2
    r1, r2 = 2.52E-2, 7.9E-2
    M_num = np.array([[j1, 0], [0, j2]])
    C_num = np.array([[c1_numvalue * r1 ** 2, -c1_numvalue * r1 * r2],
                      [-c1_numvalue * r1 * r2, c2_numvalue + c1_numvalue * r2 ** 2]])
    K_num = np.array([[k1_numvalue * r1 ** 2, -k1_numvalue * r1 * r2],
                      [-k1_numvalue * r1 * r2, k2_numvalue + k1_numvalue * r2 ** 2]])

    # Train NN
    model = train_nn(Dynpress, Iservo, M_num, C_num, K_num)

    # Generate predictions
    Y_sol = model(torch.tensor(Dynpress, dtype=torch.float32),
                  torch.tensor(Iservo, dtype=torch.float32)).detach().numpy()

    # Time values
    t_values = np.linspace(0, len(Delta) - 1, len(Delta)) / 1000

    # Plot results
    plot_results(t_values, Delta, DeltaDrum, Y_sol)


    model2_nn(run=1, k1_numvalue=1000, k2_numvalue=2000, c1_numvalue=10, c2_numvalue=20)