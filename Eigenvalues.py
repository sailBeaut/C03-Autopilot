import numpy as np

# Define a 4x4 numpy array

j1,j2,k1,k2,c1,c2,r1,r2,l = 5.4E-5, 7.97E-2, 1E5, 1E5, 1E1, 1E1, 2.52E-2, 7.9E-2, 0.5

A = np.array([[0, 0, 1, 0],[0, 0, 0, 1],[-k1*r1**2/j1, k1*r1*r2/j1, -c1*r1**2/j1,c1*r1*r2/j1],[k1*r1*r2/j2, (k1*r2**2 - k2*l**2)/j2, c1*r1*r2/j2, (c1*r2**2 - c2*l**2)/j2]])

# Calculate the eigenvalues
eigenvalues = np.linalg.eigvals(A)

# Print the eigenvalues
print("Eigenvalues of the matrix are:", eigenvalues)
