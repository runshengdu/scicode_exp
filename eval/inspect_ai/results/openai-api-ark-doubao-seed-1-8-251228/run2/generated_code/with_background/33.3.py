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
    kx0 = 2 * sqrt(3) * pi / (3 * a)
    ky0 = 4 * pi / (3 * a)
    
    # Generate k-point grid in the Brillouin zone
    kx_list = np.arange(0, kx0, delta)
    ky_list = np.arange(0, ky0, delta)
    
    total_F = 0.0 + 0.0j  # Initialize total curvature (complex)
    
    for kx in kx_list:
        for ky in ky_list:
            # Apply periodic boundary conditions for k-points
            kx_plus = kx + delta
            if kx_plus >= kx0:
                kx_plus -= kx0
            
            ky_plus = ky + delta
            if ky_plus >= ky0:
                ky_plus -= ky0
            
            # Compute Hamiltonians at four relevant k-points
            H00 = calc_hamiltonian(kx, ky, a, t1, t2, phi, m)
            H10 = calc_hamiltonian(kx_plus, ky, a, t1, t2, phi, m)
            H01 = calc_hamiltonian(kx, ky_plus, a, t1, t2, phi, m)
            H11 = calc_hamiltonian(kx_plus, ky_plus, a, t1, t2, phi, m)
            
            # Extract eigenvectors for the lower energy band
            evals00, evecs00 = np.linalg.eigh(H00)
            psi00 = evecs00[:, 0]
            
            evals10, evecs10 = np.linalg.eigh(H10)
            psi10 = evecs10[:, 0]
            
            evals01, evecs01 = np.linalg.eigh(H01)
            psi01 = evecs01[:, 0]
            
            evals11, evecs11 = np.linalg.eigh(H11)
            psi11 = evecs11[:, 0]
            
            # Calculate Ux at (kx, ky)
            overlap_x00 = np.vdot(psi00, psi10)
            abs_ov_x00 = np.abs(overlap_x00)
            ux00 = overlap_x00 / abs_ov_x00 if abs_ov_x00 > 1e-12 else 1.0
            
            # Calculate Uy at (kx_plus, ky)
            overlap_y10 = np.vdot(psi10, psi11)
            abs_ov_y10 = np.abs(overlap_y10)
            uy10 = overlap_y10 / abs_ov_y10 if abs_ov_y10 > 1e-12 else 1.0
            
            # Calculate Ux at (kx, ky_plus)
            overlap_x01 = np.vdot(psi01, psi11)
            abs_ov_x01 = np.abs(overlap_x01)
            ux01 = overlap_x01 / abs_ov_x01 if abs_ov_x01 > 1e-12 else 1.0
            
            # Calculate Uy at (kx, ky)
            overlap_y00 = np.vdot(psi00, psi01)
            abs_ov_y00 = np.abs(overlap_y00)
            uy00 = overlap_y00 / abs_ov_y00 if abs_ov_y00 > 1e-12 else 1.0
            
            # Compute product of U terms (including inverses via complex conjugation)
            product = ux00 * uy10 * np.conj(ux01) * np.conj(uy00)
            
            # Calculate curvature and accumulate total
            F = cmath.log(product)
            total_F += F
    
    # Compute Chern number by normalizing total curvature
    chern_number = np.real(total_F / (2 * pi * 1j))
    
    return chern_number



def compute_chern_number_grid(delta, a, t1, t2, N):
    '''Function to calculate the Chern numbers by sweeping the given set of parameters and returns the results along with the corresponding swept next-nearest-neighbor coupling constant and phase.
    Inputs:
    delta : float
        The grid size in kx and ky axis for discretizing the Brillouin zone.
    a : float
        The lattice spacing, i.e., the length of one side of the hexagon.
    t1 : float
        The nearest-neighbor coupling constant.
    t2 : float
        The next-nearest-neighbor coupling constant.
    N : int
        The number of sweeping grid points for both the on-site energy to next-nearest-neighbor coupling constant ratio and phase.
    Outputs:
    results: matrix of shape(N, N)
        The Chern numbers by sweeping the on-site energy to next-nearest-neighbor coupling constant ratio (m/t2) and phase (phi).
    m_values: array of length N
        The swept on-site energy to next-nearest-neighbor coupling constant ratios.
    phi_values: array of length N
        The swept phase values.
    '''
    # Generate swept parameter arrays
    m_values = np.linspace(-6, 6, N)
    phi_values = np.linspace(-pi, pi, N)
    
    # Initialize results matrix
    results = np.zeros((N, N), dtype=np.float64)
    
    # Iterate over all parameter pairs
    for i in range(N):
        m_ratio = m_values[i]
        m = m_ratio * t2  # Convert ratio to actual on-site energy
        for j in range(N):
            phi = phi_values[j]
            # Compute Chern number for current parameter set
            chern_num = compute_chern_number(delta, a, t1, t2, phi, m)
            results[i, j] = chern_num
    
    return results, m_values, phi_values
