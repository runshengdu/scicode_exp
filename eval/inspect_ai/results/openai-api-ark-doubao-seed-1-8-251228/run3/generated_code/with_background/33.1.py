import numpy as np
import cmath
from math import pi, sin, cos, sqrt



def calc_hamiltonian(kx, ky, a, t1, t2, phi, m):
    '''Function to generate the Haldane Hamiltonian with a given set of parameters.
    Inputs:
    kx : float
        The x component of the wavevector.
    ky : float
        The y component of the wavevector.
    a : float
        The lattice spacing, i.e., the length of one side of the hexagon.
    t1 : float
        The nearest-neighbor coupling constant.
    t2 : float
        The next-nearest-neighbor coupling constant.
    phi : float
        The phase ranging from -π to π.
    m : float
        The on-site energy.
    Output:
    hamiltonian : matrix of shape(2, 2)
        The Haldane Hamiltonian on a hexagonal lattice.
    '''
    # Calculate terms for d1 and d2 (nearest-neighbor contributions)
    arg1_d1 = ky * a
    arg2_d1 = (sqrt(3) / 2) * kx * a - (1 / 2) * ky * a
    arg3_d1 = (-sqrt(3) / 2) * kx * a - (1 / 2) * ky * a
    
    sum_d1 = cos(arg1_d1) + cos(arg2_d1) + cos(arg3_d1)
    d1 = t1 * sum_d1
    
    sum_d2 = sin(arg1_d1) + sin(arg2_d1) + sin(arg3_d1)
    d2 = t1 * sum_d2
    
    # Calculate terms for d0 and d3 (next-nearest-neighbor contributions)
    arg1_d0 = sqrt(3) * kx * a
    arg2_d0 = (-sqrt(3) / 2) * kx * a + (3 / 2) * ky * a
    arg3_d0 = (-sqrt(3) / 2) * kx * a - (3 / 2) * ky * a
    
    sum_d0 = cos(arg1_d0) + cos(arg2_d0) + cos(arg3_d0)
    d0 = 2 * t2 * cos(phi) * sum_d0
    
    sum_d3 = sin(arg1_d0) + sin(arg2_d0) + sin(arg3_d0)
    d3 = m - 2 * t2 * sin(phi) * sum_d3
    
    # Define identity and Pauli matrices
    I = np.eye(2, dtype=np.complex128)
    sigma1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    
    # Construct the Hamiltonian matrix
    hamiltonian = d0 * I + d1 * sigma1 + d2 * sigma2 + d3 * sigma3
    
    return hamiltonian
