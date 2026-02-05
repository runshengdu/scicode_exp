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



def crank_nicolson(sigma, kappa, T, nstep, N, L):
    '''Solve the Crank-Nicolson equation of the form A * psi(x, t+h) = B * psi(x, t)
    Input
    sigma: the sigma parameter of a Gaussian wave packet; float
    kappa: the kappa parameter of a Gaussian wave packet; float
    T: the total amount of time for the evolution in seconds; float
    nstep: the total number of time steps; int
    N: the total number of grid intervals; int
    L: the dimension of the 1D well in meters; float
    Output
    psi: the real part of the wavefunction after time T; 1D array of float with shape (N+1,)
    '''
    # Create grid points spanning the well from 0 to L
    x = np.linspace(0, L, N + 1)
    x0 = L / 2  # Center of the 1D well for initial wave packet
    
    # Initialize complex wavefunction with boundary conditions (zero at ends)
    psi = np.zeros(N + 1, dtype=np.complex128)
    # Compute initial wave packet for inner grid points
    inner_x = x[1:N]
    psi[1:N] = np.exp(-(inner_x - x0)**2 / (2 * sigma**2)) * np.exp(1j * kappa * inner_x)
    
    # Handle case with no time evolution
    if nstep == 0:
        return np.real(psi)
    
    # Calculate time step size
    h = T / nstep
    
    # Retrieve tridiagonal matrices A and B from initialization function
    A, B = init_AB(N, L, h)
    
    # Extract inner wavefunction (excluding boundaries) for time evolution
    psi_inner = psi[1:N].copy()
    M = N - 1  # Dimension of matrices A and B
    
    # Perform time evolution using Crank-Nicolson method
    if M > 0:
        # Precompute LU decomposition of A for efficient repeated solving
        lu, piv = linalg.lu_factor(A)
        for _ in range(nstep):
            # Compute right-hand side of the linear system
            b = B @ psi_inner
            # Solve for the next inner wavefunction state
            psi_inner = linalg.lu_solve((lu, piv), b)
        
        # Update the full wavefunction with evolved inner points
        psi[1:N] = psi_inner
    
    # Return the real part of the wavefunction at all grid points
    return np.real(psi)
