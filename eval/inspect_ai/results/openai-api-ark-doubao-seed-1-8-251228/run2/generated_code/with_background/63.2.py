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
