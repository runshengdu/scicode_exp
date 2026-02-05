import numpy as np

def make_IC(n):
    '''The function computes the inital condition mentioned above
    Inputs:
    n  : number of grid points, integer
    Outputs:
    v  : cell averaged approximation of initial condition, 1d array size n-1
    '''
    # Create uniform grid vertices over the domain [-π/2, π/2]
    x = np.linspace(-np.pi/2, np.pi/2, n)
    
    # Extract left and right edges of each cell
    a = x[:-1]
    b = x[1:]
    
    # Midpoint and half-width of each cell
    c = (a + b) / 2
    d = (b - a) / 2
    
    # Three-point Gauss-Legendre nodes in the reference interval [-1, 1]
    t1 = -np.sqrt(3/5)
    t2 = 0.0
    t3 = np.sqrt(3/5)
    
    # Map reference nodes to each physical cell [a, b]
    x1 = c + d * t1
    x2 = c + d * t2
    x3 = c + d * t3
    
    # Evaluate initial condition at each Gauss point
    u1 = np.where(x1 <= 0, np.sin(x1) - 1, np.sin(x1) + 1)
    u2 = np.where(x2 <= 0, np.sin(x2) - 1, np.sin(x2) + 1)
    u3 = np.where(x3 <= 0, np.sin(x3) - 1, np.sin(x3) + 1)
    
    # Compute weighted sum for three-point Gauss quadrature
    sum_weights = (5/9) * u1 + (8/9) * u2 + (5/9) * u3
    
    # Calculate cell-averaged values: (1/h) * integral ≈ sum_weights / 2
    v = sum_weights / 2
    
    return v


def LaxF(uL, uR):
    '''This function computes Lax-Fridrich numerical flux.
    Inputs: 
    uL : Cell averaged value at cell i, float
    uR : Cell averaged value at cell i+1, float
    Output: flux, float
    '''
    # Calculate flux function values at uL and uR
    fL = 0.5 * uL ** 2
    fR = 0.5 * uR ** 2
    
    # Global Lax-Friedrichs stability parameter (max |u| from initial condition)
    alpha_LF = 2.0
    
    # Compute Lax-Friedrich numerical flux using the given formula
    flux = 0.5 * (fL + fR - alpha_LF * (uR - uL))
    
    return flux



def solve(n_x, n_t, T):
    '''Inputs:
    n_x : number of spatial grids, Integer
    n_t : number of temperal grids, Integer
    T   : final time, float
    Outputs
    u1   : solution vector, 1d array of size n_x-1
    '''
    # Initialize cell-averaged initial condition
    u = make_IC(n_x)
    m = len(u)  # Number of cells, equal to n_x - 1
    
    # Calculate spatial step size (uniform grid)
    h = np.pi / (n_x - 1)
    
    # Calculate time step size
    dt = T / n_t
    
    # Time stepping loop using first-order Euler
    for _ in range(n_t):
        # Initialize flux array (m+1 fluxes for m cells)
        flux = np.zeros(m + 1)
        
        # Left boundary flux: free boundary condition (extrapolate interior state)
        flux[0] = LaxF(u[0], u[0])
        
        # Interior fluxes between adjacent cells
        flux[1:-1] = LaxF(u[:-1], u[1:])
        
        # Right boundary flux: free boundary condition (extrapolate interior state)
        flux[-1] = LaxF(u[-1], u[-1])
        
        # Compute update term from finite volume semi-discrete equation
        update = (dt / h) * (flux[1:] - flux[:-1])
        
        # Update solution vector
        u = u - update
    
    return u
