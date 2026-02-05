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
