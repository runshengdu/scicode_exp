import numpy as np

def Fermi(N_a, N_d, n_i):
    '''This function computes the Fermi levels of the n-type and p-type regions.
    Inputs:
    N_d: float, doping concentration in n-type region # cm^{-3}
    N_a: float, doping concentration in p-type region # cm^{-3}
    n_i: float, intrinsic carrier density # cm^{-3}
    Outputs:
    phi_p: float, built-in bias in p-type region (compare to E_i)
    phi_n: float, built-in bias in n-type region (compare to E_i)
    '''
    kBT = 0.0259
    phi_p = kBT * np.log(N_a / n_i)
    phi_n = kBT * np.log(N_d / n_i)
    
    return phi_p, phi_n



def depletion(N_a, N_d, n_i, e_r):
    '''This function calculates the depletion width in both n-type and p-type regions.
    Inputs:
    N_d: float, doping concentration in n-type region # cm^{-3}
    N_a: float, doping concentration in p-type region # cm^{-3}
    n_i: float, intrinsic carrier density # cm^{-3}
    e_r: float, relative permittivity
    Outputs:
    xn: float, depletion width in n-type side # cm
    xp: float, depletion width in p-type side # cm
    '''
    # Obtain built-in biases from Fermi level function
    phi_p, phi_n = Fermi(N_a, N_d, n_i)
    phi_i = phi_p + phi_n
    
    # Physical constants
    epsilon0 = 8.854e-14  # Vacuum permittivity in F/cm
    q = 1.6e-19           # Elementary charge in C
    epsilon = e_r * epsilon0  # Total permittivity
    
    # Calculate total depletion width
    xd_squared = (2 * epsilon * phi_i / q) * (N_a + N_d) / (N_a * N_d)
    xd = np.sqrt(xd_squared)
    
    # Derive n-type and p-type depletion widths using charge neutrality
    xn = xd * N_a / (N_a + N_d)
    xp = xd * N_d / (N_a + N_d)
    
    return xn, xp
