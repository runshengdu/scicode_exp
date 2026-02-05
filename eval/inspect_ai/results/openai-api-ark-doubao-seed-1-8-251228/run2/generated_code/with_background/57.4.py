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
    return x ** 2 - En


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
    f = np.asarray(f_in)
    n_points = len(f)
    u = np.zeros_like(f)
    
    if n_points == 0:
        return u
    
    # Set initial boundary condition
    u[0] = u_b
    
    if n_points == 1:
        return u
    
    # Compute the second initial point using Taylor expansion (up to second order)
    h = step
    u[1] = u[0] + h * up_b + 0.5 * h ** 2 * f[0] * u[0]
    
    # Precompute constants for efficiency
    h_sq = h ** 2
    h_sq_over_12 = h_sq / 12
    
    # Iterate using Numerov formula for remaining points
    for i in range(1, n_points - 1):
        # Calculate numerator terms
        term1 = 2 * u[i] * (1 - h_sq_over_12 * f[i])
        term2 = u[i-1] * (1 - h_sq_over_12 * f[i-1])
        term3 = h_sq * f[i] * u[i]
        numerator = term1 - term2 + term3
        
        # Calculate denominator
        denominator = 1 - h_sq_over_12 * f[i+1]
        
        # Compute next u value
        u[i+1] = numerator / denominator
    
    return u


def Solve_Schrod(x, En, u_b, up_b, step):
    '''Input
    x: coordinate x; a float or a 1D array of float
    En: energy; a float
    u_b: value of u(x) at one boundary for the Numverov function; a float
    up_b: value of the derivative of u(x) at one boundary for the Numverov function; a float
    step: the step size for the Numerov method; a float
    Output
    u_norm: normalized u(x); a float or a 1D array of float
    '''
    # Compute f(x) values at discretized points
    f_in = f_x(x, En)
    # Obtain unnormalized solution using Numerov method
    u_unnorm = Numerov(f_in, u_b, up_b, step)
    # Calculate integral of |u|² using Simpson's rule
    integral = integrate.simpson(u_unnorm ** 2, x=x)
    # Compute normalization constant
    norm_constant = 1.0 / np.sqrt(integral)
    # Normalize the solution
    u_norm = norm_constant * u_unnorm
    return u_norm



def count_sign_changes(solv_schrod):
    '''Input
    solv_schrod: a 1D array
    Output
    sign_changes: number of times of sign change occurrence; an int
    '''
    solv = np.asarray(solv_schrod)
    if len(solv) < 2:
        return 0
    consecutive_products = solv[:-1] * solv[1:]
    sign_changes = np.sum(consecutive_products < 0)
    return int(sign_changes)
