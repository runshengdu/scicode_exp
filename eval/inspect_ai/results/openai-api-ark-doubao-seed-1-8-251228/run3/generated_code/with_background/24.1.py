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
