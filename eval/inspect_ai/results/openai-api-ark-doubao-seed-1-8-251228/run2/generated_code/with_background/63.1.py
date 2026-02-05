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
