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
    # Calculate arguments for nearest-neighbor terms (a vectors)
    arg_a1 = ky * a
    arg_a2 = (sqrt(3) * a * kx) / 2 - (a * ky) / 2
    arg_a3 = (-sqrt(3) * a * kx) / 2 - (a * ky) / 2
    
    # Compute d1 and d2
    d1 = t1 * (cos(arg_a1) + cos(arg_a2) + cos(arg_a3))
    d2 = t1 * (sin(arg_a1) + sin(arg_a2) + sin(arg_a3))
    
    # Calculate arguments for next-nearest-neighbor terms (b vectors)
    arg_b1 = sqrt(3) * a * kx
    arg_b2 = (-sqrt(3) * a * kx) / 2 + (3 * a * ky) / 2
    arg_b3 = (-sqrt(3) * a * kx) / 2 - (3 * a * ky) / 2
    
    # Compute d0 and d3
    sum_cos_d0 = cos(arg_b1) + cos(arg_b2) + cos(arg_b3)
    d0 = 2 * t2 * cos(phi) * sum_cos_d0
    
    sum_sin_d3 = sin(arg_b1) + sin(arg_b2) + sin(arg_b3)
    d3 = m - 2 * t2 * sin(phi) * sum_sin_d3
    
    # Define identity and Pauli matrices
    I = np.eye(2, dtype=np.complex128)
    sigma1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    
    # Construct Hamiltonian
    hamiltonian = d0 * I + d1 * sigma1 + d2 * sigma2 + d3 * sigma3
    
    return hamiltonian
