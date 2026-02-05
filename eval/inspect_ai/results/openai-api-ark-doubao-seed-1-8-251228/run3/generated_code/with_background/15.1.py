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
    hbar = 1.0545718e-34  # Reduced Planck's constant in Js

    # Calculate grid spacing
    if N == 0:
        a = 0.0
    else:
        a = L / N

    # Compute complex coefficient C
    if a == 0:
        C = 0j
    else:
        C = (1j * hbar) / (4 * m * a ** 2)

    # Calculate diagonal and off-diagonal elements for A and B
    a1 = 1 + 2 * h * C
    a2 = -h * C
    b1 = 1 - 2 * h * C
    b2 = h * C

    # Dimension of matrices A and B
    M = N - 1

    # Handle case with no internal grid points to time evolve
    if M <= 0:
        A = np.array([]).reshape(0, 0)
        B = np.array([]).reshape(0, 0)
        return A, B

    # Construct symmetric tridiagonal matrix A
    main_diag_A = np.full(M, a1)
    off_diag_A = np.full(M - 1, a2)
    A = np.diag(main_diag_A) + np.diag(off_diag_A, k=1) + np.diag(off_diag_A, k=-1)

    # Construct symmetric tridiagonal matrix B
    main_diag_B = np.full(M, b1)
    off_diag_B = np.full(M - 1, b2)
    B = np.diag(main_diag_B) + np.diag(off_diag_B, k=1) + np.diag(off_diag_B, k=-1)

    return A, B
