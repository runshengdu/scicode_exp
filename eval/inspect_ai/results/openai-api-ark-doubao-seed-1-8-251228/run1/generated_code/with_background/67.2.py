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
    delta_l = abs(l1 - l2)
    sum_l = l1 + l2
    term1 = np.exp(-q * d * delta_l)
    term2 = alpha * np.exp(-q * d * sum_l)
    form_factor = term1 + term2
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
    if q == 0.0:
        return 0.0 + 0.0j
    
    omega_prime = omega + 1j * gamma
    # Calculate B = q²/(2m) = e_F (q/k_F)²
    B = e_F * (q / k_F) ** 2
    # Calculate v_F q / k_F
    qv_over_k = q * v_F / k_F
    # Calculate denominator terms
    term1_num = omega_prime + B
    term1_den = qv_over_k
    z1 = term1_num / term1_den
    term2_num = omega_prime - B
    z2 = term2_num / term1_den
    
    # Calculate the argument for the square root
    sqrt_arg1 = 1.0 - z1 ** 2
    sqrt_arg2 = 1.0 - z2 ** 2
    
    # Compute square roots, using numpy's sqrt which handles complex numbers
    sqrt1 = np.sqrt(sqrt_arg1)
    sqrt2 = np.sqrt(sqrt_arg2)
    
    # Calculate the terms for the integral result
    term_a = np.log(z1 + sqrt1) if np.imag(z1 + sqrt1) != 0 else np.log(np.real(z1 + sqrt1))
    term_b = np.log(z2 + sqrt2) if np.imag(z2 + sqrt2) != 0 else np.log(np.real(z2 + sqrt2))
    
    # Combine terms
    integral_result = (term_a - term_b) / (2 * np.pi)
    
    # Multiply by n_eff / e_F to get D0
    D0 = (n_eff / e_F) * integral_result
    
    return D0
