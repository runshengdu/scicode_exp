import numpy as np


def matrix_elements(lambda_in, lambda_b, n1, n2):
    '''Calculates the phase shift and the A/B/C/D matrix factors for a given wavelength.
    Input:
    lambda_in (float): Wavelength of the incident light in nanometers.
    lambda_b (float): Resonant wavelength in nanometers.
    n1 (float): Refractive index of the first material.
    n2 (float): Refractive index of the second material.
    Output:
    matrix (2 by 2 numpy array containing 4 complex numbers): Matrix used in the calculation of the transmission coefficient.
    '''
    # Calculate phase shift phi
    phi = (np.pi * lambda_b) / (2 * lambda_in)
    
    # Compute coefficient terms for A
    term1_a = (n1 + n2) ** 2 / (4 * n1 * n2)
    term2_a = (n1 - n2) ** 2 / (4 * n1 * n2)
    A = term1_a * np.exp(-2j * phi) - term2_a
    
    # Compute B
    coeff_b = (n2 ** 2 - n1 ** 2) / (4 * n1 * n2)
    B = coeff_b * (1 - np.exp(2j * phi))
    
    # Compute conjugate terms for C and D
    C = np.conj(B)
    D = np.conj(A)
    
    # Construct and return the 2x2 matrix
    matrix = np.array([[A, B], [C, D]])
    
    return matrix



def get_theta(A, D):
    '''Calculates the angle theta used in the calculation of the transmission coefficient.
    If the value of (A + D) / 2 is greater than 1, keep the real part of theta as np.pi.
    Input:
    A (complex): Matrix factor from the calculation of phase shift and matrix factors.
    D (complex): Matrix factor from the calculation of phase shift and matrix factors.
    Output:
    theta (complex): Angle used in the calculation of the transmission coefficient.
    '''
    x = (A + D) / 2
    if np.abs(x) <= 1:
        theta = np.arccos(x)
    else:
        # For |x| > 1, theta is pi + i*alpha where cos(theta) = x
        # cos(pi + i*alpha) = -cosh(alpha) = x => cosh(alpha) = -x (if x < -1)
        # Since x can't be >1 per problem context, we handle x < -1
        alpha = np.arccosh(-x) if np.isreal(x) and x < -1 else np.arccosh(np.abs(x))
        theta = np.pi + 1j * alpha
    return theta



def R_coefficient(lambda_in, lambda_b, n1, n2, N):
    '''Calculates the total reflection coefficient for a given number of layer pairs.
    If theta is complex, uses hyperbolic sine functions in the calculation.
    Input:
    lambda_in (float): Wavelength of the incident light in nanometers.
    lambda_b (float): Resonant wavelength in nanometers.
    n1 (float): Refractive index of the first material.
    n2 (float): Refractive index of the second material.
    N (int): Number of pairs of layers.
    Output:
    R (float): Total reflection coefficient.
    '''
    # Retrieve the propagation matrix
    mat = matrix_elements(lambda_in, lambda_b, n1, n2)
    A = mat[0, 0]
    D = mat[1, 1]
    C = mat[1, 0]
    
    # Calculate the pseudo-angle theta
    theta = get_theta(A, D)
    
    # Compute squared magnitude of C
    abs_C_squared = np.abs(C) ** 2
    
    # Compute squared magnitude of the sine ratio term
    sin_theta = np.sin(theta)
    sin_N_theta = np.sin(N * theta)
    abs_sin_ratio_squared = (np.abs(sin_theta / sin_N_theta)) ** 2
    
    # Calculate reflection coefficient
    R = abs_C_squared / (abs_C_squared + abs_sin_ratio_squared)
    
    return R
