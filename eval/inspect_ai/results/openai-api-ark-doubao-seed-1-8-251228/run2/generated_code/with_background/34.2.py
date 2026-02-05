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
    kT = 0.0259
    phi_p = kT * np.log(N_a / n_i)
    phi_n = kT * np.log(N_d / n_i)
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
    # Calculate built-in potential using Fermi function
    phi_p, phi_n = Fermi(N_a, N_d, n_i)
    phi_i = phi_p + phi_n
    
    # Physical constants
    epsilon0 = 8.854e-14  # Vacuum permittivity in F/cm
    q = 1.6e-19           # Electron charge in C
    epsilon = e_r * epsilon0
    
    # Compute total depletion width
    term = (2 * epsilon * phi_i) / q * (N_a + N_d) / (N_a * N_d)
    x_d = np.sqrt(term)
    
    # Calculate individual depletion widths using charge neutrality
    xn = x_d * N_a / (N_a + N_d)
    xp = x_d * N_d / (N_a + N_d)
    
    return xn, xp
