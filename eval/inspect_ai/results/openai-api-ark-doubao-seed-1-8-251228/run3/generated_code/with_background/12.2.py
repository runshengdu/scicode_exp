from scipy import integrate
from scipy import optimize
import numpy as np

def f_Schrod(energy, l, r_grid):
    '''Input 
    energy: a float
    l: angular momentum quantum number; an int
    r_grid: the radial grid; a 1D array of float
    Output
    f_r: a 1D array of float 
    '''
    Z = 1
    r_B = 0.529  # Bohr radius in Angstroms
    E0 = 13.6    # Rydberg energy in eV
    
    # Convert to dimensionless variables
    x = r_grid / r_B
    eps = energy / E0
    
    # Calculate dimensionless expression from the differential equation
    f_x = l * (l + 1) / (x ** 2) - 2 * Z / x - eps
    
    # Convert to dimensional f(r) in units of 1/Angstrom²
    f_r = f_x / (r_B ** 2)
    
    return f_r



def Numerov(f_in, u_at_0, up_at_0, step):
    '''Given precomputed function f(r), solve the differential equation u''(r) = f(r)*u(r)
    using the Numerov method.
    Inputs:
    - f_in: input function f(r); a 1D array of float representing the function values at discretized points.
    - u_at_0: the value of u at r = 0; a float.
    - up_at_0: the derivative of u at r = 0; a float.
    - step: step size; a float.
    Output:
    - u: the integration results at each point in the radial grid; a 1D array of float.
    '''

    
    n = len(f_in)
    u = np.zeros_like(f_in)
    u[0] = u_at_0
    
    if n == 1:
        return u
    
    # Compute the second point using Taylor expansion up to u''(0)
    h = step
    h_sq = h ** 2
    u_second_deriv_0 = f_in[0] * u[0]
    u[1] = u[0] + h * up_at_0 + 0.5 * h_sq * u_second_deriv_0
    
    if n == 2:
        return u
    
    # Precompute constants for Numerov recursion
    h_sq_over_12 = h_sq / 12
    five_hso12 = 5 * h_sq_over_12
    
    # Iterate using the Numerov formula for third-order linear ODE
    for i in range(1, n - 1):
        term1 = 2 * u[i] * (1 + five_hso12 * f_in[i])
        term2 = u[i-1] * (1 - h_sq_over_12 * f_in[i-1])
        denominator = 1 - h_sq_over_12 * f_in[i+1]
        u[i+1] = (term1 - term2) / denominator
    
    return u
