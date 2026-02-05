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
