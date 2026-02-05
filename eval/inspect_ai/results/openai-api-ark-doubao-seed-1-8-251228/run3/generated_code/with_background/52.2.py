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
