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



def solve(CFL, T, dt, alpha):
    '''Inputs:
    CFL : Courant-Friedrichs-Lewy condition number
    T   : Max time, float
    dt  : Time interval, float
    alpha : diffusive coefficient , float
    Outputs:
    u   : solution, array of float
    '''
    # Compute initial dx from CFL condition for diffusion
    dx = np.sqrt(alpha * dt / CFL)
    
    # Set up spatial grid (assuming domain [0, 1] as default)
    domain_length = 1.0
    n = int(domain_length / dx) + 1
    dx = domain_length / (n - 1)  # Adjust dx to fit domain exactly
    x = np.linspace(0, domain_length, n)
    
    # Initial condition: Gaussian centered at 0.5 with standard deviation 0.1
    u = np.exp(-((x - 0.5) ** 2) / (0.1 ** 2))
    
    # Calculate number of time steps
    nt = int(T / dt)
    
    for _ in range(nt):
        # Step 1: Half-step with diffusion operator f0 (forward Euler)
        u_half_f0 = np.copy(u)
        for target in range(n):
            u_xx = second_diff(target, u, dx)
            u_half_f0[target] = u[target] + (dt / 2) * alpha * u_xx
        
        # Step 2: Full-step with reaction operator f1 (forward Euler)
        u_full_f1 = np.copy(u_half_f0)
        for target in range(n):
            # Example linear reaction term: R(u) = -u (decay, adjust if needed)
            reaction_term = -u_half_f0[target]
            u_full_f1[target] = u_half_f0[target] + dt * reaction_term
        
        # Step 3: Half-step with diffusion operator f0 again (forward Euler)
        u = np.copy(u_full_f1)
        for target in range(n):
            u_xx = second_diff(target, u_full_f1, dx)
            u[target] = u_full_f1[target] + (dt / 2) * alpha * u_xx
    
    return u
