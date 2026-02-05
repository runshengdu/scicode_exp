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
