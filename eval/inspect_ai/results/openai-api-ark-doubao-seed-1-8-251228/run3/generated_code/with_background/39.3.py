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
    phi = np.pi * lambda_b / (2 * lambda_in)
    
    # Calculate components for matrix element A
    term1_A = (n1 + n2) ** 2 / (4 * n1 * n2)
    term2_A = (n1 - n2) ** 2 / (4 * n1 * n2)
    A = term1_A * np.exp(-2 * 1j * phi) - term2_A
    
    # Calculate components for matrix element B
    term_B = (n2 ** 2 - n1 ** 2) / (4 * n1 * n2)
    B = term_B * (1 - np.exp(2 * 1j * phi))
    
    # Calculate conjugate elements C and D
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
    x_real = np.real(x)
    
    if x_real < -1:
        alpha = np.arccosh(-x_real)
        theta = np.pi + 1j * alpha
    else:
        theta = np.arccos(x_real)
    
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
    # Retrieve the one-period transfer matrix
    matrix = matrix_elements(lambda_in, lambda_b, n1, n2)
    A = matrix[0][0]
    C = matrix[1][0]
    D = matrix[1][1]
    
    # Calculate the real value x = (A + D) / 2
    x = np.real((A + D) / 2)
    
    # Compute theta using the provided get_theta function
    theta = get_theta(A, D)
    
    # Calculate squared magnitude of C
    abs_C_sq = np.abs(C) ** 2
    
    # Determine the appropriate term based on the magnitude of x
    if np.abs(x) > 1:
        alpha = np.imag(theta)
        sinh_alpha = np.sinh(alpha)
        sinh_N_alpha = np.sinh(N * alpha)
        term = np.abs(sinh_alpha / sinh_N_alpha) ** 2
    else:
        sin_theta = np.sin(theta)
        sin_N_theta = np.sin(N * theta)
        term = np.abs(sin_theta / sin_N_theta) ** 2
    
    # Compute the reflection coefficient
    R = abs_C_sq / (abs_C_sq + term)
    
    return R
