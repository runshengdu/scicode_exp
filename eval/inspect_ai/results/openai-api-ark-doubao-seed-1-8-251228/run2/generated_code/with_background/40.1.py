import numpy as np



def second_diff(target, u, dx):
    '''Inputs:
    target : Target cell index, int
    u      : Approximated solution value, array of floats with minimum length of 5
    Outputs:
    deriv  : Second order derivative of the given target cell, float
    '''
    n = len(u)
    u_curr = u[target]
    
    # Determine left value (handle left boundary ghost cell)
    if target == 0:
        u_prev = u[0]
    else:
        u_prev = u[target - 1]
    
    # Determine right value (handle right boundary ghost cell)
    if target == n - 1:
        u_next = u[-1]
    else:
        u_next = u[target + 1]
    
    # Calculate second order derivative using the centered scheme
    deriv = (u_next - 2 * u_curr + u_prev) / (dx ** 2)
    
    return deriv
