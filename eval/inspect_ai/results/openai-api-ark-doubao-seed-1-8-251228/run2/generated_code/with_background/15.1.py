import numpy as np
from scipy import linalg, sparse



def init_AB(N, L, h):
    '''Initialize the matrices A and B
    Input
    N: the number of grid intervals; int
    L: the dimension of the 1D well; float
    h: the size of each time step in seconds; float
    Output
    A,B: A and B matrices; 2D arrays of dimension N-1 by N-1 where each element is a float
    '''
    # Physical constants
    m = 9.109e-31  # Electron mass in kg
    hbar = 1.054571817e-34  # Reduced Planck's constant in Js
    
    # Calculate grid spacing
    a = L / N
    
    # Compute complex coefficient C
    C = (1j * hbar) / (4 * m * a**2)
    
    # Calculate diagonal and off-diagonal elements for A and B
    a1 = 1 + 2 * h * C
    a2 = -h * C
    b1 = 1 - 2 * h * C
    b2 = h * C
    
    # Dimension of the matrices
    dim = N - 1
    
    # Handle edge case where there are no interior points
    if dim <= 0:
        A = np.array([]).reshape(0, 0)
        B = np.array([]).reshape(0, 0)
        return A, B
    
    # Create diagonal arrays
    diag_A = np.full(dim, a1)
    offdiag_A = np.full(dim - 1, a2)
    
    diag_B = np.full(dim, b1)
    offdiag_B = np.full(dim - 1, b2)
    
    # Construct symmetric tridiagonal matrices
    A = np.diag(diag_A) + np.diag(offdiag_A, k=1) + np.diag(offdiag_A, k=-1)
    B = np.diag(diag_B) + np.diag(offdiag_B, k=1) + np.diag(offdiag_B, k=-1)
    
    return A, B
