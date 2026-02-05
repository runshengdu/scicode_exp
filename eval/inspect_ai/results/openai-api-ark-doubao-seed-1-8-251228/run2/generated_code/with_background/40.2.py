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



def Strang_splitting(u, dt, dx, alpha):
    n = len(u)
    # Step 1: Half-step with f0 (diffusion operator) using explicit Euler
    u_half_f0 = np.copy(u)
    for target in range(n):
        # Calculate second derivative using the pre-defined second_diff function
        u_xx = second_diff(target, u, dx)
        # Update solution for half-step of diffusion
        u_half_f0[target] = u[target] + (dt / 2) * alpha * u_xx
    
    # Step 2: Full step with f1 (operator placeholder - adjust if f1 is defined)
    # For this implementation, we assume f1 is a placeholder (e.g., zero operator)
    u_full_f1 = np.copy(u_half_f0)
    
    # Step 3: Half-step with f0 (diffusion operator) again using explicit Euler
    u_check = np.copy(u_full_f1)
    for target in range(n):
        # Calculate second derivative using the updated solution from step 2
        u_xx = second_diff(target, u_full_f1, dx)
        # Update solution for final half-step of diffusion
        u_check[target] = u_full_f1[target] + (dt / 2) * alpha * u_xx
    
    return u_check
