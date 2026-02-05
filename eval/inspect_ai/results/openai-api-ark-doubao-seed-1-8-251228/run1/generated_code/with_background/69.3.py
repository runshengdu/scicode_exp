import numpy as np

def f_V(q, d, bg_eps, l1, l2):
    '''Write down the form factor f(q;l1,l2)
    Input
    q, in-plane momentum, float in the unit of inverse angstrom
    d, layer spacing, float in the unit of angstrom
    bg_eps: LEG dielectric constant, float, dimensionless
    l1,l2: layer number where z = l*d, integer
    Output
    form_factor: form factor, float
    '''
    alpha = (bg_eps - 1) / (bg_eps + 1)
    distance_direct = d * abs(l1 - l2)
    distance_image = d * (l1 + l2)
    term_direct = np.exp(-q * distance_direct)
    term_image = alpha * np.exp(-q * distance_image)
    form_factor = term_direct + term_image
    return form_factor



def D_2DEG(q, omega, gamma, n_eff, e_F, k_F, v_F):
    '''Write down the exact form of density-density correlation function
    Input
    q, in-plane momentum, float in the unit of inverse angstrom
    omega, energy, real part, float in the unit of meV
    gamma, energy, imaginary part, float in the unit of meV
    n_eff, electron density, float in the unit of per square angstrom
    e_F, Fermi energy, float in the unit of meV
    k_F, Fermi momentum, float in the unit of inverse angstrom
    v_F, hbar * Fermi velocity, float in the unit of meV times angstrom
    Output
    D0: density-density correlation function, complex array in the unit of per square angstrom per meV
    '''
    if q == 0:
        return 0.0 + 0.0j
    
    omega_complex = omega + gamma * 1j
    vq = v_F * q
    numerator = omega_complex
    denominator = np.sqrt(omega_complex ** 2 - vq ** 2)
    
    # Handle the case when denominator is zero (measure zero in practice)
    if denominator == 0:
        term = 0.0
    else:
        term = numerator / denominator
    
    # Compute the correlation function using the analytical form
    D0 = (2 * n_eff / e_F) * (term - 1.0)
    
    return D0



def D_cal(D0, q, d, bg_eps, N):
    '''Calculate the matrix form of density-density correlation function D(l1,l2)
    Input
    D0, density-density correlation function, complex array in the unit of per square angstrom per meV
    q, in-plane momentum, float in the unit of inverse angstrom
    d, layer spacing, float in the unit of angstrom
    bg_eps: LEG dielectric constant, float, dimensionless
    N: matrix dimension, integer
    Output
    D: NxN complex matrix, in the unit of per square angstrom per meV
    '''
    # Create meshgrid for layer indices (i,j) where i is row index, j is column index
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    
    # Calculate alpha from dielectric constant
    alpha = (bg_eps - 1) / (bg_eps + 1)
    
    # Compute direct and image distances for all layer pairs
    distance_direct = d * np.abs(i - j)
    distance_image = d * (i + j)
    
    # Calculate form factor matrix elements
    term_direct = np.exp(-q * distance_direct)
    term_image = alpha * np.exp(-q * distance_image)
    F = term_direct + term_image
    F = F.astype(np.complex128)  # Ensure complex type compatibility with D0
    
    # Construct identity matrix
    identity = np.eye(N, dtype=np.complex128)
    
    # Solve Dyson equation: D = D0 * (I - D0 * F)^{-1}
    inv_matrix = np.linalg.inv(identity - D0 * F)
    D = D0 * inv_matrix
    
    return D
