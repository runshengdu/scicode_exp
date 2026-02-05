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
    direct_term = np.exp(-q * d * abs(l1 - l2))
    alpha = (bg_eps - 1) / (bg_eps + 1)
    image_term = alpha * np.exp(-q * d * (l1 + l2))
    form_factor = direct_term + image_term
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
    # Handle q=0 case where the numerator of the integral is zero
    if np.isclose(q, 0.0):
        return 0.0 + 0.0j
    
    q_critical = 2 * k_F
    x = q / q_critical
    y = (omega + 1j * gamma) / (q * v_F)
    
    if q <= q_critical:
        sqrt_term = np.sqrt(1 - x**2)
        arctan_argument = sqrt_term / y
        arctan_value = np.arctan(arctan_argument)
        correction_term = (y / sqrt_term) * arctan_value
    else:
        sqrt_term = np.sqrt(x**2 - 1)
        log_numerator = y + sqrt_term
        log_denominator = y - sqrt_term
        log_argument = log_numerator / log_denominator
        log_value = np.log(log_argument)
        correction_term = (y / (2 * sqrt_term)) * log_value
    
    formula = 1 - correction_term
    D0 = (n_eff / e_F) * formula
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
    # Handle q=0 case where D0 is zero to avoid division by zero
    if np.isclose(q, 0.0):
        return np.zeros((N, N), dtype=np.complex128)
    
    # Construct the form factor matrix F
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    direct_term = np.exp(-q * d * np.abs(i - j))
    alpha = (bg_eps - 1) / (bg_eps + 1)
    image_term = alpha * np.exp(-q * d * (i + j))
    F = direct_term + image_term
    
    # Calculate 2D Coulomb interaction V_q using given vacuum dielectric constant
    epsilon0 = 55.26349406  # Vacuum dielectric constant in e² eV⁻¹ μm⁻¹
    e2_over_2epsilon0_eV_um = 1 / (2 * epsilon0)
    # Convert eV·μm to meV·Å: 1 eV=1000 meV, 1 μm=1e4 Å
    e2_over_2epsilon0_meV_ang = e2_over_2epsilon0_eV_um * 1000 * 1e4
    V_q = e2_over_2epsilon0_meV_ang / q  # V_q in meV·Å²
    
    # Construct interaction matrix V = V_q * F
    V_matrix = V_q * F
    
    # Construct non-interacting D0 matrix (diagonal matrix)
    D0_matrix = np.eye(N, dtype=np.complex128) * D0
    
    # Solve Dyson equation: D = (I - D0 @ V)⁻¹ @ D0
    M = np.eye(N, dtype=np.complex128) - D0_matrix @ V_matrix
    D = np.linalg.inv(M) @ D0_matrix
    
    return D
