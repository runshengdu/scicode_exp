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
