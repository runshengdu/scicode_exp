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
