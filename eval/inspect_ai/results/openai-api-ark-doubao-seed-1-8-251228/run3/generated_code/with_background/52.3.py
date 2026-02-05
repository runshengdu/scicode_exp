import numpy as np
from scipy import integrate, optimize

def Schroed_deriv(y, r, l, En):
    '''Calculate the derivative of y given r, l and En
    Input 
    y=[u,u'], an list of float where u is the wave function at r, u' is the first derivative of u at r
    r: radius, float
    l: angular momentum quantum number, int
    En: energy, float
    Output
    Schroed: dy/dr=[u',u''] , an 1D array of float where u is the wave function at r, u' is the first derivative of u at r, u'' is the second derivative of u at r
    '''
    u, u_prime = y
    if r == 0:
        # Safeguard against division by zero (solver typically avoids r=0 in practice)
        u_double_prime = 0.0
    else:
        # Calculate second derivative using the dimensionless radial equation (Z=1)
        u_double_prime = (l * (l + 1) / (r ** 2) - 2.0 / r - En) * u
    return np.array([u_prime, u_double_prime])


def SolveSchroedinger(y0, En, l, R):
    '''Integrate the derivative of y within a certain radius
    Input 
    y0: Initial guess for function and derivative, list of floats: [u0, u0']
    En: energy, float
    l:  angular momentum quantum number, int
    R:  an 1d array (linespace) of radius (float)
    Output
    ur: the integration result, array of floats
    '''
    # Determine if the radius array is increasing or decreasing
    is_increasing = R[0] < R[-1]
    
    # Create integration array starting from large r to small r
    r_integrate = R[::-1] if is_increasing else R
    
    # Integrate the ODE using the provided derivative function
    y_result = integrate.odeint(Schroed_deriv, y0, r_integrate, args=(l, En))
    
    # Extract u values in the original radius array order
    if is_increasing:
        u_vals = y_result[::-1, 0]
    else:
        u_vals = y_result[:, 0]
    
    # Calculate the normalization integral using Simpson's rule
    norm_squared = integrate.simpson(u_vals ** 2, x=R)
    
    # Normalize the wavefunction
    ur = u_vals / np.sqrt(norm_squared)
    
    return ur



def Shoot(En, R, l, y0):
    '''Extrapolate u(0) based on results from SolveSchroedinger function
    Input 
    y0: Initial guess for function and derivative, list of floats: [u0, u0']
    En: energy, float
    R: an 1D array of (logspace) of radius; each element is a float
    l: angular momentum quantum number, int
    Output 
    f_at_0: Extrapolate u(0), float
    '''
    # Obtain the normalized radial wavefunction from the solver
    ur = SolveSchroedinger(y0, En, l, R)
    
    # Extract the first two points from the radial grid and corresponding wavefunction values
    r0, r1 = R[0], R[1]
    u0, u1 = ur[0], ur[1]
    
    # Scale the wavefunction values by dividing by r^l
    s0 = u0 / (r0 ** l)
    s1 = u1 / (r1 ** l)
    
    # Perform linear extrapolation to r=0
    if r1 == r0:
        # Avoid division by zero (logspace ensures distinct points, but handle edge case)
        f_at_0 = s0
    else:
        f_at_0 = (s0 * r1 - s1 * r0) / (r1 - r0)
    
    return f_at_0
