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
    # Calculate log-transformed min and max prices
    p_min = np.log(min_price)
    p_max = np.log(max_price)
    
    # Create log price grid (column vector)
    p = np.linspace(p_min, p_max, price_step).reshape(-1, 1)
    
    # Calculate price step size
    dp = (p_max - p_min) / (price_step - 1) if price_step > 1 else 0.0
    
    # Create time grid from 0 (initial time) to 1 (maturity time) as column vector
    T = np.linspace(0, 1, time_step).reshape(-1, 1)
    
    # Calculate time step size
    dt = 1.0 / (time_step - 1) if time_step > 1 else 0.0
    
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
    
    # Temporal boundary condition (maturity time, last column)
    s = np.exp(p[0])  # Convert log prices to actual stock prices
    payoff = np.maximum(s - strike, 0.0)
    V[:, -1] = payoff
    
    # Lower price boundary (smallest stock price, first row)
    V[0, :] = 0.0
    
    # Upper price boundary (largest stock price, last row)
    s_max = np.exp(p[0, -1])  # Maximum stock price from log grid
    time_remaining = 1.0 - T[0]  # Time until maturity for each time point
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
    
    # Handle cases with no interior points
    if M <= 0:
        return sparse.csr_matrix((M, M))
    
    # Calculate coefficients a, b, c from finite difference equations
    term1 = r - 0.5 * sig ** 2
    a = (dt / 2) * (term1 / dp - (sig ** 2) / (dp ** 2))
    b = 1 + dt * ((sig ** 2) / (dp ** 2) + r)
    c = - (dt / 2) * (term1 / dp + (sig ** 2) / (dp ** 2))
    
    # Prepare diagonals and their respective offsets
    diagonals = []
    offsets = []
    
    # Main diagonal (coefficient b for all interior points)
    main_diag = np.full(M, b)
    diagonals.append(main_diag)
    offsets.append(0)
    
    # Subdiagonal and superdiagonal for matrices with more than 1 interior point
    if M > 1:
        sub_diag = np.full(M - 1, a)  # Coefficient a for previous interior point
        super_diag = np.full(M - 1, c)  # Coefficient c for next interior point
        diagonals.extend([sub_diag, super_diag])
        offsets.extend([-1, 1])
    
    # Construct and return the sparse tridiagonal matrix
    D = sparse.diags(diagonals, offsets, shape=(M, M))
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
    # Handle cases with no interior points or only one time step
    if N_p < 3 or N_t < 2:
        return V
    
    M = N_p - 2
    
    # Calculate coefficients a and c needed for boundary terms
    term1 = r - 0.5 * sig ** 2
    if dp == 0:
        a = 0.0
        c = 0.0
    else:
        a = (dt / 2) * (term1 / dp - (sig ** 2) / (dp ** 2))
        c = - (dt / 2) * (term1 / dp + (sig ** 2) / (dp ** 2))
    
    # Iterate backwards from maturity to initial time
    for n in range(N_t - 2, -1, -1):
        # Get the interior values at the next time step (closer to maturity)
        V_interior_next = V[1:-1, n+1]
        
        # Construct the boundary term vector B
        B = np.zeros(M)
        B[0] = a * V[0, n]
        B[-1] = c * V[-1, n]
        
        # Compute the right-hand side of the linear system
        rhs = V_interior_next - B
        
        # Solve the sparse linear system D * V_interior_current = rhs
        V_interior_current = spsolve(D, rhs)
        
        # Update the interior values at the current time step
        V[1:-1, n] = V_interior_current
    
    return V
