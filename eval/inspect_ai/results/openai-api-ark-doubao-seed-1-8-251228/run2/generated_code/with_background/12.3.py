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
    # Constants from background
    r_B = 0.529  # Bohr radius in Å
    E0 = 13.6    # Rydberg energy in eV
    Z = 1        # Atomic number as specified
    
    # Calculate dimensionless variables
    x = r_grid / r_B
    eps = energy / E0
    
    # Compute f(x) in dimensionless form
    f_x = l * (l + 1) / (x ** 2) - 2 * Z / x - eps
    
    # Convert to f(r) for the original variable r
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
    f_in = np.asarray(f_in)
    n = len(f_in)
    u = np.zeros(n, dtype=np.float64)
    u[0] = u_at_0
    
    if n == 1:
        return u
    
    h = step
    # Compute u[1] using Taylor expansion up to O(h^3)
    u1 = u_at_0 + h * up_at_0 + 0.5 * h**2 * f_in[0] * u_at_0
    u[1] = u1
    
    if n == 2:
        return u
    
    # Iterate using direct Numerov recursion for remaining points
    for i in range(1, n - 1):
        numerator = 24 * u[i] - 12 * u[i-1] + (h**2) * (10 * f_in[i] * u[i] + f_in[i-1] * u[i-1])
        denominator = 12 - (h**2) * f_in[i+1]
        u[i+1] = numerator / denominator
    
    return u



def compute_Schrod(energy, r_grid, l):
    '''Input 
    energy: a float
    l: angular momentum quantum number; an int
    r_grid: the radial grid; a 1D array of float
    Output
    ur_norm: normalized wavefunction u(x) at x = r
    '''
    # Reverse the radial grid to integrate from the largest radius inward
    r_reversed = r_grid[::-1]
    # Compute the f(r) values for the reversed grid
    f_reversed = f_Schrod(energy, l, r_reversed)
    # Calculate the step size as specified (r_grid[0] - r_grid[1])
    step = r_grid[0] - r_grid[1]
    # Solve the differential equation using the Numerov method
    u_reversed = Numerov(f_reversed, u_at_0=0.0, up_at_0=-1e-7, step=step)
    # Reverse the solution to match the original grid order
    u_original = u_reversed[::-1]
    # Compute the normalization integral using Simpson's rule
    integral = integrate.simpson(u_original ** 2, x=r_grid)
    # Normalize the wavefunction
    ur_norm = u_original / np.sqrt(integral)
    
    return ur_norm
