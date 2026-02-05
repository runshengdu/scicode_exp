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
