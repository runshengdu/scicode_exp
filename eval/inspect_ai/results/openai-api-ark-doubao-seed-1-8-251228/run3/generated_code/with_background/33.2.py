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



def compute_chern_number(delta, a, t1, t2, phi, m):
    '''Function to compute the Chern number with a given set of parameters.
    Inputs:
    delta : float
        The grid size in kx and ky axis for discretizing the Brillouin zone.
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
    chern_number : float
        The Chern number, a real number that should be close to an integer. The imaginary part is cropped out due to the negligible magnitude.
    '''
    # Define Brillouin zone boundaries
    kx0 = 2 * np.sqrt(3) * np.pi / (3 * a)
    ky0 = 4 * np.pi / (3 * a)
    
    # Generate k-point grids
    kx_list = np.arange(0, kx0, delta)
    ky_list = np.arange(0, ky0, delta)
    
    sum_F = 0.0 + 0.0j  # Initialize sum of curvatures
    
    for kx in kx_list:
        for ky in ky_list:
            # Get eigenvector for current k-point (lower band)
            h = calc_hamiltonian(kx, ky, a, t1, t2, phi, m)
            eig_vals, eig_vecs = np.linalg.eigh(h)
            n1 = eig_vecs[:, 0]
            
            # Get eigenvector for kx+delta (mod kx0)
            kx2 = (kx + delta) % kx0
            h2 = calc_hamiltonian(kx2, ky, a, t1, t2, phi, m)
            eig_vals2, eig_vecs2 = np.linalg.eigh(h2)
            n2 = eig_vecs2[:, 0]
            
            # Get eigenvector for ky+delta (mod ky0)
            ky3 = (ky + delta) % ky0
            h3 = calc_hamiltonian(kx, ky3, a, t1, t2, phi, m)
            eig_vals3, eig_vecs3 = np.linalg.eigh(h3)
            n3 = eig_vecs3[:, 0]
            
            # Get eigenvector for kx+delta, ky+delta (mod both boundaries)
            kx4 = (kx + delta) % kx0
            ky4 = (ky + delta) % ky0
            h4 = calc_hamiltonian(kx4, ky4, a, t1, t2, phi, m)
            eig_vals4, eig_vecs4 = np.linalg.eigh(h4)
            n4 = eig_vecs4[:, 0]
            
            # Compute overlaps between eigenvectors
            o12 = np.vdot(n1, n2)
            o13 = np.vdot(n1, n3)
            o24 = np.vdot(n2, n4)
            o34 = np.vdot(n3, n4)
            
            # Calculate U terms (avoid division by zero)
            abs_o12 = np.abs(o12)
            ux = o12 / abs_o12 if abs_o12 > 1e-12 else 1.0
            
            abs_o13 = np.abs(o13)
            uy = o13 / abs_o13 if abs_o13 > 1e-12 else 1.0
            
            abs_o24 = np.abs(o24)
            uy_x = o24 / abs_o24 if abs_o24 > 1e-12 else 1.0
            
            abs_o34 = np.abs(o34)
            ux_y = o34 / abs_o34 if abs_o34 > 1e-12 else 1.0
            
            # Compute the product for curvature
            prod = ux * uy_x * np.conj(ux_y) * np.conj(uy)
            
            # Calculate curvature contribution
            f_xy = np.log(prod)
            sum_F += f_xy
    
    # Compute Chern number from total curvature
    chern_number = (sum_F / (2j * np.pi)).real
    
    return chern_number
