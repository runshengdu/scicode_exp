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
    alpha = (bg_eps - 1.0) / (bg_eps + 1.0)
    exponent_direct = q * d * abs(l1 - l2)
    term_direct = np.exp(-exponent_direct)
    exponent_image = q * d * (l1 + l2)
    term_image = np.exp(-exponent_image)
    form_factor = term_direct + alpha * term_image
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
    # Handle q=0 case to avoid division by zero
    if abs(q) < 1e-12:
        return 0.0 + 0.0j
    
    c = e_F * (q ** 2) / (k_F ** 2)
    u_max = 2 * e_F * q / k_F
    
    C1 = omega + 1j * gamma + c
    C2 = omega + 1j * gamma - c
    
    # Calculate terms from integral evaluation
    sqrt_C1_sq = np.sqrt(C1 ** 2)
    sqrt_C1_sq_u = np.sqrt(C1 ** 2 - u_max ** 2)
    term1 = sqrt_C1_sq - sqrt_C1_sq_u
    
    sqrt_C2_sq = np.sqrt(C2 ** 2)
    sqrt_C2_sq_u = np.sqrt(C2 ** 2 - u_max ** 2)
    term2 = sqrt_C2_sq - sqrt_C2_sq_u
    
    # Compute integral over radial momentum
    denominator = 4 * e_F ** 2 * q ** 2
    integral_r = (k_F ** 4) / denominator * (term1 - term2)
    
    # Final density-density correlation function
    D0 = (1.0 / np.pi) * integral_r
    
    return D0




def D_cal(D0, q, d, bg_eps, N):
    '''Calculate the matrix form of density-density correlation function D(l1,l2)
    Input
    D0, density-density correlation function, complex array in the unit of per square angstrom per meV
    q, in-plane momentum, float in the unit of inverse angstrom
    d, layer spacing, float in the unit of angstrom
    bg_eps: LEG dielectric constant, float
    N: matrix dimension, integer
    Output
    D: NxN complex matrix, in the unit of per square angstrom per meV
    '''
    # Handle q=0 case to avoid division by zero and infinite V_q
    if abs(q) < 1e-12:
        return np.zeros((N, N), dtype=np.complex128)
    
    # Create meshgrid for layer indices (i,j) corresponding to (l1,l2)
    i_mat, j_mat = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    
    # Compute form factor matrix F using vectorized operations
    alpha = (bg_eps - 1.0) / (bg_eps + 1.0)
    exponent_direct = q * d * np.abs(i_mat - j_mat)
    term_direct = np.exp(-exponent_direct)
    exponent_image = q * d * (i_mat + j_mat)
    term_image = np.exp(-exponent_image)
    F = term_direct + alpha * term_image
    
    # Calculate 2D Coulomb interaction V_q in meV·Å²
    epsilon0 = 55.26349406  # Vacuum dielectric constant in e² eV⁻¹ μm⁻¹
    constant = 1.0e7 / (2.0 * np.pi * epsilon0)
    V_q = constant / (bg_eps * q)
    
    # Construct M matrix: I - D0 * V_q * F
    M = np.eye(N, dtype=np.complex128) - D0 * V_q * F
    
    # Compute inverse of M and scale by D0 to get the final matrix D
    inv_M = np.linalg.inv(M)
    D = D0 * inv_M
    
    return D
