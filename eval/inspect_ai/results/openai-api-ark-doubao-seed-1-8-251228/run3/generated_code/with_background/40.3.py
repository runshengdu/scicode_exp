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


def Strang_splitting(u, dt, dx, alpha):
    '''Inputs:
    u : solution, array of float
    dt: time interval , float
    dx: sptial interval, float
    alpha: diffusive coefficient, float
    Outputs:
    u : solution, array of float
    '''

    n = len(u)
    
    # Step 1: Half-step with diffusion operator f0 using dt/2
    u_hat = np.zeros_like(u)
    for i in range(n):
        # Calculate second derivative using the provided centered scheme with ghost cells
        left = u[i-1] if i > 0 else u[0]
        right = u[i+1] if i < n-1 else u[-1]
        current = u[i]
        u_xx = (right - 2 * current + left) / (dx ** 2)
        # Update for half-time step of diffusion
        u_hat[i] = current + (dt / 2) * alpha * u_xx
    
    # Step 2: Full-step with advection operator f1 using dt (first-order upwind scheme)
    u_tilde = np.zeros_like(u_hat)
    for i in range(n):
        current = u_hat[i]
        # Compute first derivative with upwind scheme and ghost cell boundary handling
        if current >= 0:
            # Upwind from left (use ghost cell at boundary)
            left_val = u_hat[i-1] if i > 0 else u_hat[0]
            u_x = (current - left_val) / dx
        else:
            # Upwind from right (use ghost cell at boundary)
            right_val = u_hat[i+1] if i < n-1 else u_hat[-1]
            u_x = (right_val - current) / dx
        # Update for full-time step of advection (Burgers' equation advection term)
        u_tilde[i] = current - dt * current * u_x
    
    # Step 3: Half-step with diffusion operator f0 using dt/2
    u_check = np.zeros_like(u_tilde)
    for i in range(n):
        # Calculate second derivative using the provided centered scheme with ghost cells
        left = u_tilde[i-1] if i > 0 else u_tilde[0]
        right = u_tilde[i+1] if i < n-1 else u_tilde[-1]
        current = u_tilde[i]
        u_xx = (right - 2 * current + left) / (dx ** 2)
        # Update for half-time step of diffusion
        u_check[i] = current + (dt / 2) * alpha * u_xx
    
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


    # Calculate spatial step size from CFL condition
    dx = np.sqrt((alpha * dt) / CFL)
    
    # Set up spatial grid and initial condition (Gaussian pulse)
    n_points = 100
    x = np.linspace(0, (n_points - 1) * dx, n_points)
    mean_x = x.mean()
    std_dev = 0.1 * x.max()  # 10% of domain length for Gaussian width
    u = np.exp(-(x - mean_x) ** 2 / (2 * std_dev ** 2))
    
    t = 0.0
    while t < T:
        # Adjust time step for final iteration if needed
        current_dt = dt if (t + dt) <= T else (T - t)
        n = len(u)
        
        # Step 1: Half-step with diffusion operator f0 using forward Euler
        u_hat = np.zeros_like(u)
        for i in range(n):
            # Compute second derivative with ghost cell boundary handling
            left = u[i-1] if i > 0 else u[0]
            right = u[i+1] if i < n-1 else u[-1]
            current = u[i]
            u_xx = (right - 2 * current + left) / (dx ** 2)
            f0 = alpha * u_xx
            u_hat[i] = current + (current_dt / 2) * f0
        
        # Step 2: Full step with reaction operator f1 (Fisher-KPP reaction term)
        u_tilde = u_hat + current_dt * u_hat * (1 - u_hat)
        
        # Step 3: Half-step with diffusion operator f0 using forward Euler
        u_new = np.zeros_like(u_tilde)
        for i in range(n):
            # Compute second derivative with ghost cell boundary handling
            left = u_tilde[i-1] if i > 0 else u_tilde[0]
            right = u_tilde[i+1] if i < n-1 else u_tilde[-1]
            current = u_tilde[i]
            u_xx = (right - 2 * current + left) / (dx ** 2)
            f0 = alpha * u_xx
            u_new[i] = current + (current_dt / 2) * f0
        
        # Update solution and time
        u = u_new
        t += current_dt
    
    return u
