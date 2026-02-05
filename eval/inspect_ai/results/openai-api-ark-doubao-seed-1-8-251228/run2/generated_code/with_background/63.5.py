import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def initialize_grid(price_step, time_step, strike, max_price, min_price):
    '''Initializes the grid for pricing a European call option.
    Inputs:
    price_step: The number of steps or intervals in the price direction. (int)
    time_step: The number of steps or intervals in the time direction. (int)
    strike: The strike price of the European call option. (float)
    max_price: we can't compute infinity as bound, so set a max bound. 5 * strike price is generous (float)
    min_price: avoiding 0 as a bound due to numerical instability, (1/5) * strike price is generous (float)
    Outputs:
    p: An array containing the grid points for prices. It is calculated using np.linspace function between p_min and p_max.  shape: price_step * 1
    dp: The spacing between adjacent price grid points. (float)
    T: An array containing the grid points for time. It is calculated using np.linspace function between 0 and 1. shape: time_step * 1
    dt: The spacing between adjacent time grid points. (float)
    '''
    # Calculate log bounds for stock price grid
    p_min = np.log(min_price)
    p_max = np.log(max_price)
    
    # Create log stock price grid (p) and reshape to (price_step, 1)
    p = np.linspace(p_min, p_max, price_step).reshape(-1, 1)
    
    # Calculate spacing between adjacent log price points
    dp = p[1, 0] - p[0, 0] if price_step > 1 else 0.0
    
    # Create time grid from 0 (current time) to 1 (maturity) and reshape to (time_step, 1)
    T = np.linspace(0, 1, time_step).reshape(-1, 1)
    
    # Calculate spacing between adjacent time points
    dt = T[1, 0] - T[0, 0] if time_step > 1 else 0.0
    
    return p, dp, T, dt


def apply_boundary_conditions(N_p, N_t, p, T, strike, r, sig):
    '''Applies the boundary conditions to the grid.
    Inputs:
    N_p: The number of grid points in the price direction. = price_step (int)
    N_t: The number of grid points in the time direction. = time_step (int)
    p: An array containing the grid points for prices. (shape = 1 * N_p , (float))
    T: An array containing the grid points for time. (shape = 1 * N_t , (float))
    strike: The strike price of the European call option. (float)
    r: The risk-free interest rate. (float)
    sig: The volatility of the underlying stock. (float)
    Outputs:
    V: A 2D array representing the grid for the option's value after applying boundary conditions. Shape: N_p x N_t where N_p is number of price grid, and N_t is number of time grid
    '''
    # Initialize the value grid with zeros
    V = np.zeros((N_p, N_t))
    
    # Temporal boundary condition: final time step (maturity)
    n_final = N_t - 1
    s = np.exp(p)
    V[:, n_final] = np.maximum(s - strike, 0.0).flatten()
    
    # Lower price boundary condition (minimum stock price, all time steps)
    V[0, :] = 0.0
    
    # Upper price boundary condition (maximum stock price, all time steps)
    s_max = np.exp(p[0, -1])
    time_remaining = 1 - T.flatten()
    upper_bound_values = s_max - strike * np.exp(-r * time_remaining)
    V[-1, :] = upper_bound_values
    
    return V


def construct_matrix(N_p, dp, dt, r, sig):
    '''Constructs the tri-diagonal matrix for the finite difference method.
    Inputs:
    N_p: The number of grid points in the price direction. (int)
    dp: The spacing between adjacent price grid points. (float)
    dt: The spacing between adjacent time grid points. (float)
    r: The risk-free interest rate. (float)
    sig: The volatility of the underlying asset. (float)
    Outputs:
    D: The tri-diagonal matrix constructed for the finite difference method. Shape: (N_p-2)x(N_p-2) where N_p is number of price grid, and N_t is number of time grid minus 2 due to boundary conditions
    '''
    M = N_p - 2
    if M <= 0:
        return sparse.csr_matrix((0, 0))
    
    # Calculate coefficients a, b, c using the given formulas
    term1 = (r - 0.5 * sig**2) / dp
    term2 = sig**2 / (dp**2)
    
    a = (dt / 2) * (term1 - term2)
    b = 1 + dt * (term2 + r)
    c = - (dt / 2) * (term1 + term2)
    
    # Create the three diagonals for the tri-diagonal matrix
    sub_diag = np.full(M - 1, a) if M > 1 else np.array([])
    main_diag = np.full(M, b)
    super_diag = np.full(M - 1, c) if M > 1 else np.array([])
    
    # Construct the sparse tri-diagonal matrix
    D = sparse.diags([sub_diag, main_diag, super_diag], [-1, 0, 1], shape=(M, M))
    
    return D


def forward_iteration(V, D, N_p, N_t, r, sig, dp, dt):
    '''Performs the forward iteration to solve for option prices at earlier times.
    Inputs:
    V: A 2D array representing the grid for the option's value at different times and prices. Shape: N_p x N_t (float)
    D: The tri-diagonal matrix constructed for the finite difference method. Shape: (N_t-2) x (N_t-2) (float)
    N_p: The number of grid points in the price direction. (int)
    N_t: The number of grid points in the time direction. (int)
    r: The risk-free interest rate. (float)
    sig: The volatility of the underlying asset. (float)
    dp: The spacing between adjacent price grid points. (float)
    dt: The spacing between adjacent time grid points. (float)
    Outputs:
    V: Updated option value grid after performing forward iteration. Shape: N_p x N_t where N_p is number of price grid, and N_t is number of time grid
    '''
    # Handle edge cases where no iteration is needed
    if N_p <= 2 or N_t <= 1:
        return V
    
    M = N_p - 2  # Number of interior price points
    
    # Calculate coefficients a and c using same formulas as construct_matrix
    term1 = (r - 0.5 * sig**2) / dp
    term2 = sig**2 / (dp ** 2)
    a = (dt / 2) * (term1 - term2)
    c = - (dt / 2) * (term1 + term2)
    
    # Iterate backwards from second-to-last time step to initial time step
    for n in range(N_t-2, -1, -1):
        # Extract interior points from the next (future) time step
        V_next_int = V[1:-1, n+1]
        
        # Construct boundary contribution vector B
        B = np.zeros(M)
        B[0] += a * V[0, n]    # Lower boundary contribution for first interior point
        B[-1] += c * V[-1, n]  # Upper boundary contribution for last interior point
        
        # Compute right-hand side of the linear system
        rhs = V_next_int - B
        
        # Solve for interior points at current time step
        V_int_n = spsolve(D, rhs)
        
        # Update the value grid with computed interior points
        V[1:-1, n] = V_int_n
    
    return V



def price_option(price_step, time_step, strike, r, sig, max_price, min_price):
    '''Prices a European call option using the finite difference method.
    Inputs:
    price_step: The number of steps or intervals in the price direction. = N_p (int)
    time_step: The number of steps or intervals in the time direction. = N_t (int)
    strike: The strike price of the European call option. (float)
    r: The risk-free interest rate. (float)
    sig: The volatility of the underlying asset. (float)
    max_price: we can't compute infinity as bound, so set a max bound. 5 * strike price is generous (float)
    min_price: avoiding 0 as a bound due to numerical instability, (1/5) * strike price is generous (float)
    Outputs:
    V: A 2D array representing the grid for the option's value. Shape: N_p x N_t where N_p is number of price grid, and N_t is number of time grid
    '''
    # Initialize price and time grids
    p, dp, T, dt = initialize_grid(price_step, time_step, strike, max_price, min_price)
    
    # Apply boundary conditions to get initial value grid
    V = apply_boundary_conditions(price_step, time_step, p, T, strike, r, sig)
    
    # Construct the recursive matrix D
    D = construct_matrix(price_step, dp, dt, r, sig)
    
    # Perform forward iteration to fill in option prices for all time steps
    V = forward_iteration(V, D, price_step, time_step, r, sig, dp, dt)
    
    return V
