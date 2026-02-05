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
    # Calculate log price boundaries (p = log s)
    p_min = np.log(min_price)
    p_max = np.log(max_price)
    
    # Create log price grid as column vector with specified number of points
    p = np.linspace(p_min, p_max, num=price_step).reshape(-1, 1)
    
    # Calculate price step size (dp)
    if price_step == 1:
        dp = 0.0
    else:
        dp = p[1, 0] - p[0, 0]  # Even spacing from linspace
    
    # Create time grid (0 to maturity normalized to 1) as column vector
    T = np.linspace(0, 1, num=time_step).reshape(-1, 1)
    
    # Calculate time step size (dt)
    if time_step == 1:
        dt = 0.0
    else:
        dt = T[1, 0] - T[0, 0]  # Even spacing from linspace
    
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
    # Initialize option value grid with zeros
    V = np.zeros((N_p, N_t))
    
    # Extract 1D arrays for price and time grids
    p_flat = p.flatten()
    T_flat = T.flatten()
    
    # --------------------------
    # Price Boundary Conditions
    # --------------------------
    # Boundary 1: Minimum price (p_min) - option price is 0 for all time (call option)
    V[0, :] = 0.0
    
    # Boundary 2: Maximum price (p_max) - option price approximates s_max - K*e^{-r(T-t)} for all time
    p_max = p_flat[-1]
    s_max = np.exp(p_max)
    time_remaining = 1 - T_flat  # Time left until maturity (normalized time)
    V[-1, :] = s_max - strike * np.exp(-r * time_remaining)
    
    # --------------------------
    # Temporal Boundary Condition (Expiration Time)
    # --------------------------
    # Payoff at maturity: max(s - strike, 0) for call option (resolves problem statement typo conflict)
    s_values = np.exp(p_flat)
    V[:, -1] = np.maximum(s_values - strike, 0.0)
    
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
    
    # Handle edge case where there are no inner points
    if M <= 0:
        return sparse.csr_matrix((max(M, 0), max(M, 0)))
    
    # Precompute squared terms to avoid redundant calculations
    sig_sq = sig ** 2
    dp_sq = dp ** 2
    r_minus_half_sig_sq = r - 0.5 * sig_sq
    
    # Calculate coefficients a, b, c using the given formulas
    a = (dt / 2) * (r_minus_half_sig_sq / dp - sig_sq / dp_sq)
    b = 1 + dt * (sig_sq / dp_sq + r)
    c = - (dt / 2) * (r_minus_half_sig_sq / dp + sig_sq / dp_sq)
    
    # Create the diagonals for the tridiagonal matrix
    sub_diagonal = [a] * (M - 1)  # Below main diagonal (offset -1)
    main_diagonal = [b] * M        # Main diagonal (offset 0)
    super_diagonal = [c] * (M - 1) # Above main diagonal (offset 1)
    
    # Construct the sparse tridiagonal matrix and convert to CSR format for efficient multiplication
    D = sparse.diags(
        diagonals=[sub_diagonal, main_diagonal, super_diagonal],
        offsets=[-1, 0, 1],
        shape=(M, M)
    ).tocsr()
    
    return D



def forward_iteration(V, D, N_p, N_t, r, sig, dp, dt):
    '''Performs the forward iteration to solve for option prices at earlier times.
    Inputs:
    V: A 2D array representing the grid for the option's value at different times and prices. Shape: N_p x N_t (float)
    D: The tri-diagonal matrix constructed for the finite difference method. Shape: (N_p-2) x (N_p-2) (float)
    N_p: The number of grid points in the price direction. (int)
    N_t: The number of grid points in the time direction. (int)
    r: The risk-free interest rate. (float)
    sig: The volatility of the underlying asset. (float)
    dp: The spacing between adjacent price grid points. (float)
    dt: The spacing between adjacent time grid points. (float)
    Outputs:
    V: Updated option value grid after performing forward iteration. Shape: N_p x N_t where N_p is number of price grid, and N_t is number of time grid
    '''
    # Calculate coefficients a and c for boundary term contributions
    if dp == 0:
        a = 0.0
        c = 0.0
    else:
        sig_sq = sig ** 2
        dp_sq = dp ** 2
        r_minus_half_sig_sq = r - 0.5 * sig_sq
        a = (dt / 2) * (r_minus_half_sig_sq / dp - sig_sq / dp_sq)
        c = - (dt / 2) * (r_minus_half_sig_sq / dp + sig_sq / dp_sq)
    
    # Iterate backwards from maturity (n = N_t-2) to initial time (n = 0)
    for n in range(N_t - 2, -1, -1):
        M = N_p - 2
        if M <= 0:
            continue  # No inner price points to update
        
        # Extract inner option values from the next time step (n+1)
        V_next_inner = V[1:-1, n+1].flatten()
        
        # Construct boundary contribution vector
        B = np.zeros(M)
        B[0] = a * V[0, n]
        B[-1] = c * V[-1, n]
        
        # Formulate right-hand side of linear system
        rhs = V_next_inner - B
        
        # Solve sparse linear system for current time step's inner values
        V_inner_n = spsolve(D, rhs)
        
        # Update the inner points in the option value grid
        V[1:-1, n] = V_inner_n
    
    return V
