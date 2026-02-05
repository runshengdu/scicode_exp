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
    # Calculate time step
    h = T / nstep if nstep != 0 else 0.0
    
    # Create grid points spanning the well
    x = np.linspace(0, L, N + 1)
    x0 = L / 2  # Center of the 1D well
    
    # Initialize Gaussian wave packet centered at x0
    psi_initial = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * kappa * x)
    # Enforce boundary conditions (wavefunction zero at endpoints)
    psi_initial[0] = 0.0
    psi_initial[-1] = 0.0
    
    # Extract interior points for time evolution (exclude endpoints)
    psi_int = psi_initial[1:-1]
    
    # Proceed with time evolution if there are interior points
    if N >= 1:
        A, B = init_AB(N, L, h)
        
        # Only perform evolution if there are non-trivial matrices
        if A.size > 0:
            # Precompute LU decomposition for efficient repeated solving
            lu, piv = linalg.lu_factor(A)
            
            # Iterate over each time step
            for _ in range(nstep):
                rhs = B @ psi_int
                psi_int = linalg.lu_solve((lu, piv), rhs)
    
    # Construct full wavefunction array with boundary conditions
    psi_full = np.zeros(N + 1, dtype=np.complex128)
    if N >= 1:
        psi_full[1:-1] = psi_int
    
    # Return the real part of the final wavefunction
    return np.real(psi_full)
