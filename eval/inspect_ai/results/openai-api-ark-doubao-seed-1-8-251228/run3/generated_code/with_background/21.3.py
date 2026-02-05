import numpy as np

def m_eff(x, m0):
    '''Calculates the effective mass of GaAlAs for a given aluminum mole fraction x.
    Input:
    x (float): Aluminum mole fraction in GaAlAs.
    m0 (float): electron rest mass (can be reduced to 1 as default).
    Output:
    mr (float): Effective mass of GaAlAs.
    '''
    # Calculate individual effective masses based on composition x
    m_e = (0.0637 + 0.083 * x) * m0
    m_lh = (0.087 + 0.063 * x) * m0
    m_hh = (0.50 + 0.29 * x) * m0
    
    # Compute light and heavy hole relative effective masses
    m_r_lh = 1 / (1 / m_e + 1 / m_lh)
    m_r_hh = 1 / (1 / m_e + 1 / m_hh)
    
    # Calculate the sum of (m_r)^(3/2) terms
    term_lh = m_r_lh ** (3/2)
    term_hh = m_r_hh ** (3/2)
    sum_terms = term_lh + term_hh
    
    # Solve for the total density of states effective mass
    mr = sum_terms ** (2/3)
    
    return mr


def alpha_eff(lambda_i, x, C):
    '''Calculates the effective absorption coefficient of AlxGa1-xAs.
    Input:
    lambda (float): Wavelength of the incident light (nm).
    x (float): Aluminum composition in the AlxGa1-xAs alloy.
    C (float): Optional scaling factor for the absorption coefficient. Default is 1.
    Returns:
    Output (float): Effective absorption coefficient in m^-1.
    '''
    # Fundamental constants
    c = 3e8  # Vacuum speed of light in m/s
    hbar = 1.0546e-34  # Reduced Planck constant in J·s
    e = 1.6022e-19  # Electron charge in C (1 eV = e Joules)
    
    # Convert wavelength from nanometers to meters
    lambda_m = lambda_i * 1e-9
    
    # Calculate angular frequency
    omega = 2 * np.pi * c / lambda_m
    
    # Calculate photon energy in Joules
    hbar_omega = hbar * omega
    
    # Calculate band gap energy in eV, then convert to Joules
    E_g_eV = 1.424 + 1.247 * x
    E_g_J = E_g_eV * e
    
    # Compute energy difference, ensure non-negative to avoid imaginary values
    delta_E = hbar_omega - E_g_J
    delta_E = max(delta_E, 0.0)
    
    # Retrieve DOS relative effective mass
    m_r = m_eff(x, m0=1)
    
    # Calculate effective absorption coefficient
    alpha_x = (C * m_r * np.sqrt(delta_E)) / omega
    
    return alpha_x



def alpha(lambda_i, x, lambda0, alpha0):
    '''Computes the absorption coefficient for given wavelength and Al composition,
    normalized by the absorption coefficient at a reference wavelength for pure GaAs.
    Input:
    lambda_i (float): Wavelength of the incident light (nm).
    x (float): Aluminum composition in the AlxGa1-xAs alloy.
    lambda0 (float): Reference wavelength (nm) for pure GaAs (x=0).
    alpha0 (float): Absorption coefficient at the reference wavelength for pure GaAs.
    Output:
    alpha_final (float): Normalized absorption coefficient in m^-1.
    '''
    # Fundamental constants
    c = 3e8  # Vacuum speed of light in m/s
    hbar = 1.0546e-34  # Reduced Planck constant in J·s
    e = 1.6022e-19  # Electron charge in C
    
    # Convert wavelengths from nanometers to meters
    lambda_m = lambda_i * 1e-9
    lambda0_m = lambda0 * 1e-9
    
    # Calculate angular frequencies
    omega = 2 * np.pi * c / lambda_m
    omega0 = 2 * np.pi * c / lambda0_m
    
    # Calculate photon energies in Joules
    hbar_omega = hbar * omega
    hbar_omega0 = hbar * omega0
    
    # Calculate band gap energies in eV, then convert to Joules
    # For AlxGa1-xAs
    E_g_x_eV = 1.424 + 1.247 * x
    E_g_x_J = E_g_x_eV * e
    # For pure GaAs (x=0)
    E_g_GaAs_eV = 1.424
    E_g_GaAs_J = E_g_GaAs_eV * e
    
    # Compute energy differences, ensure non-negative to avoid imaginary values
    delta_E_x = max(hbar_omega - E_g_x_J, 0.0)
    delta_E0 = max(hbar_omega0 - E_g_GaAs_J, 0.0)
    
    # Retrieve DOS relative effective masses
    m_r_x = m_eff(x, m0=1)
    m_r_GaAs = m_eff(0, m0=1)
    
    # Calculate the ratio using the given formula
    numerator = omega0 * m_r_x * np.sqrt(delta_E_x)
    denominator = omega * m_r_GaAs * np.sqrt(delta_E0)
    
    # Compute final absorption coefficient, handle division by zero edge case
    if denominator == 0:
        alpha_final = 0.0
    else:
        alpha_final = alpha0 * (numerator / denominator)
    
    return alpha_final
