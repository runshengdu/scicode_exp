import numpy as np



def second_diff(target, u, dx):
    '''Inputs:
    target : Target cell index, int
    u      : Approximated solution value, array of floats with minimum length of 5
    dx     : Spatial interval, float
    Outputs:
    deriv  : Second order derivative of the given target cell, float
    '''
    n = len(u)
    # Get left neighbor value (handle boundary with ghost cell)
    left = u[target - 1] if target > 0 else u[0]
    # Get right neighbor value (handle boundary with ghost cell)
    right = u[target + 1] if target < n - 1 else u[-1]
    current = u[target]
    # Calculate second derivative using centered scheme
    deriv = (right - 2 * current + left) / (dx ** 2)
    return deriv
