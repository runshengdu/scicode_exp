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
    direct_term = np.exp(-q * d * np.abs(l1 - l2))
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
    # Calculate minimum p for integral limits
    p_min = np.maximum(0.0, k_F - q)
    # Complex frequency with broadening
    omega_prime = omega + 1j * gamma
    
    # Compute key terms for the form factor
    C = e_F * q**2 / k_F**2 - omega_prime
    u_max = 2 * e_F * q / k_F
    u_min = 2 * e_F * q * p_min / k_F**2
    
    # Calculate square root terms
    sqrt_c_u_max = np.sqrt(C**2 - u_max**2)
    sqrt_c_u_min = np.sqrt(C**2 - u_min**2)
    sqrt_term = sqrt_c_u_max - sqrt_c_u_min
    
    # Calculate the coefficient based on input parameters
    coefficient = np.pi * n_eff**2 / (2 * e_F**2 * q**2)
    
    # Final correlation function
    D0 = sqrt_term * coefficient
    return D0
