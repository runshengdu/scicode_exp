import numpy as np
from scipy import integrate, optimize

def f_x(x, En):
    '''Return the value of f(x) with energy En
    Input
    x: coordinate x; a float or a 1D array of float
    En: energy; a float
    Output
    f_x: the value of f(x); a float or a 1D array of float
    '''
    return x**2 - En


def Numerov(f_in, u_b, up_b, step):
    '''Given precomputed function f(x), solve the differential equation u''(x) = f(x)*u(x)
    using the Numerov method.
    Inputs:
    - f_in: input function f(x); a 1D array of float representing the function values at discretized points
    - u_b: the value of u at boundary; a float
    - up_b: the derivative of u at boundary; a float
    - step: step size; a float.
    Output:
    - u: u(x); a 1D array of float representing the solution.
    '''
    n = len(f_in)
    u = np.zeros_like(f_in)
    u[0] = u_b
    
    if n == 1:
        return u
    
    # Compute the second initial value using Taylor expansion
    h = step
    u1 = u_b + h * up_b + 0.5 * h**2 * f_in[0] * u_b
    u[1] = u1
    
    # Precompute common terms for efficiency
    h_sq = h ** 2
    h_sq_over_12 = h_sq / 12
    
    # Iterate using Numerov method for remaining points
    for i in range(2, n):
        # Extract f values at relevant points
        f_prev_prev = f_in[i-2]
        f_prev = f_in[i-1]
        f_curr = f_in[i]
        
        # Extract u values at relevant points
        u_prev_prev = u[i-2]
        u_prev = u[i-1]
        
        # Calculate terms for Numerov formula
        term1 = 2 * u_prev * (1 - h_sq_over_12 * f_prev)
        term2 = u_prev_prev * (1 - h_sq_over_12 * f_prev_prev)
        term3 = h_sq * f_prev * u_prev
        
        numerator = term1 - term2 + term3
        denominator = 1 - h_sq_over_12 * f_curr
        
        # Compute current u value
        u[i] = numerator / denominator
    
    return u


def Solve_Schrod(x, En, u_b, up_b, step):
    '''Input
    x: coordinate x; a float or a 1D array of float
    En: energy; a float
    u_b: value of u at boundary; a float
    up_b: the derivative of u at boundary; a float
    step: the step size for the Numerov method; a float
    Output
    u_norm: normalized u(x); a float or a 1D array of float
    '''
    # Handle scalar input conversion
    x_arr = np.asarray(x)
    is_scalar = x_arr.ndim == 0
    if is_scalar:
        x_arr = x_arr.reshape(1)
    
    # Compute the f(x) values at discretized points
    f_in = f_x(x_arr, En)
    
    # Solve the ODE using Numerov method
    u = Numerov(f_in, u_b, up_b, step)
    
    # Calculate normalization integral using Simpson's rule
    norm_sq = integrate.simpson(u ** 2, x_arr)
    
    # Normalize the solution
    u_norm = u / np.sqrt(norm_sq)
    
    # Convert back to scalar if input was scalar
    if is_scalar:
        return float(u_norm[0])
    else:
        return u_norm



def count_sign_changes(solv_schrod):
    '''Input
    solv_schrod: a 1D array
    Output
    sign_changes: number of times of sign change occurrence; an int
    '''
    arr = np.asarray(solv_schrod)
    if len(arr) < 2:
        return 0
    # Calculate product of consecutive elements
    consecutive_products = arr[:-1] * arr[1:]
    # Count number of negative products (indicating sign changes)
    sign_changes = np.sum(consecutive_products < 0)
    return int(sign_changes)
