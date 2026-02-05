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
    # Calculate second derivative using the dimensionless Schrodinger equation (Z=1)
    u_double_prime = (l * (l + 1) / (r ** 2) - 2.0 / r - En) * u
    # Return the first-order system derivative as a 1D array
    return np.array([u_prime, u_double_prime])



def SolveSchroedinger(y0, En, l, R):
    '''Integrate the derivative of y within a certain radius
    Input 
    y0: Initial guess for function and derivative, list of floats: [u0, u0']
    En: energy, float
    l:  angular momentum quantum number, int
    R:  an 1d array (linespace) of radius (float)
    Output
    ur: the integration result, float
    '''
    # Determine integration order to start from large r
    if R[0] < R[-1]:
        # Reverse R to integrate from largest to smallest radius
        r_integrate = R[::-1]
        # Perform integration
        y_sol = integrate.odeint(Schroed_deriv, y0, r_integrate, args=(l, En))
        # Reverse solution to match original R order
        ur_before_norm = y_sol[:, 0][::-1]
    else:
        # R is already decreasing, integrate directly
        y_sol = integrate.odeint(Schroed_deriv, y0, R, args=(l, En))
        ur_before_norm = y_sol[:, 0]
    
    # Normalize using Simpson's rule
    integral = integrate.simpson(ur_before_norm ** 2, x=R)
    ur_normalized = ur_before_norm / np.sqrt(integral)
    
    return ur_normalized
