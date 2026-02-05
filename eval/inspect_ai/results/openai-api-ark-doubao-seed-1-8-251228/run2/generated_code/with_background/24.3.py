import numpy as np

def make_IC(n):
    '''The function computes the inital condition mentioned above
    Inputs:
    n  : number of grid points, integer
    Outputs:
    v  : cell averaged approximation of initial condition, 1d array size n-1
    '''
    # Create grid vertices spanning the domain [-π/2, π/2]
    x = np.linspace(-np.pi/2, np.pi/2, n)
    
    # Three-point Gauss-Legendre quadrature points and weights on [-1, 1]
    t = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    weights = np.array([5/9, 8/9, 5/9])
    
    # Extract left and right edges of each cell
    left_edges = x[:-1]
    right_edges = x[1:]
    
    # Transform Gauss points from [-1,1] to each cell [left_edges[i], right_edges[i]]
    cell_widths = right_edges - left_edges
    cell_midpoints = (left_edges + right_edges) / 2
    gauss_points = cell_midpoints[:, np.newaxis] + (cell_widths[:, np.newaxis] / 2) * t
    
    # Evaluate initial condition at transformed Gauss points
    u_vals = np.where(gauss_points <= 0, np.sin(gauss_points) - 1, np.sin(gauss_points) + 1)
    
    # Compute cell averages using Gauss quadrature
    cell_averages = 0.5 * np.sum(weights * u_vals, axis=1)
    
    return cell_averages


def LaxF(uL, uR):
    '''This function computes Lax-Fridrich numerical flux.
    Inputs: 
    uL : Cell averaged value at cell i, float
    uR : Cell averaged value at cell i+1, float
    Output: flux, float
    '''
    alpha_LF = 2.0
    # Compute flux function at left and right states
    fL = 0.5 * uL ** 2
    fR = 0.5 * uR ** 2
    # Calculate Lax-Friedrichs flux using the given formula
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
    
    # Calculate time step size
    dt = T / n_t
    
    # Calculate uniform cell width (domain length is π, n_x-1 cells)
    h = np.pi / (n_x - 1)
    
    # Time stepping loop with explicit Euler
    for _ in range(n_t):
        # Create extended solution with ghost cells for free boundary conditions
        extended_u = np.concatenate([[u[0]], u, [u[-1]]])
        
        # Extract left and right states for each flux face
        uL = extended_u[:-1]
        uR = extended_u[1:]
        
        # Compute all numerical fluxes using Lax-Friedrichs scheme
        fluxes = LaxF(uL, uR)
        
        # Compute time derivative from semi-discrete finite volume equation
        du_dt = (fluxes[:-1] - fluxes[1:]) / h
        
        # Update solution with Euler step
        u = u + dt * du_dt
    
    u1 = u
    return u1
