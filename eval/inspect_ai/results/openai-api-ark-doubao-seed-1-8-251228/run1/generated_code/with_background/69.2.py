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
