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
