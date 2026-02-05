import numpy as np

def Fermi(N_A, N_D, n_i):
    '''This function computes the Fermi levels of the n-type and p-type regions.
    Inputs:
    N_A: float, doping concentration in p-type region # cm^{-3}
    N_D: float, doping concentration in n-type region # cm^{-3}
    n_i: float, intrinsic carrier density # cm^{-3}
    Outputs:
    phi_p: float, built-in bias in p-type region (compare to E_i)
    phi_n: float, built-in bias in n-type region (compare to E_i)
    '''
    kT = 0.0259  # Thermal potential at room temperature in volts
    phi_p = kT * np.log(N_A / n_i)
    phi_n = kT * np.log(N_D / n_i)
    return phi_p, phi_n


def capacitance(xi, A, N_A, N_D, n_i, es, V0):
    '''Calculates the capacitance of a p-i-n diode.
    Input:
    xi (float): Width of the intrinsic region (μm).
    A (float): Detector Area (μm^2).
    N_A (float): Doping concentration of the p-type region (cm^-3).
    N_D (float): Doping concentration of the n-type region (cm^-3).
    n_i: float, intrinsic carrier density of the material # cm^{-3}
    es (float): Relative permittivity.
    V0 (float): Applied voltage to the p-i-n diode (V).
    Output:
    C (float): Capacitance of the p-i-n diode (F).
    '''
    # Calculate Fermi level positions using the provided Fermi function
    phi_p, phi_n = Fermi(N_A, N_D, n_i)
    phi_b = phi_p + phi_n  # Total built-in potential
    
    # Physical constants
    epsilon0 = 8.854e-12  # Vacuum permittivity in F/m
    q = 1.6e-19           # Electron charge in C
    
    # Calculate permittivity of the material
    epsilon = epsilon0 * es
    
    # Convert units to SI (meters and square meters)
    xi_m = xi * 1e-6                  # Convert μm to meters
    A_m2 = A * (1e-6) ** 2            # Convert μm² to m²
    
    # Compute terms for the denominator
    term1 = xi_m ** 2
    term2 = (2 * epsilon / q) * (-V0 + phi_b) * ((N_A + N_D) / (N_A * N_D)) * 1e-6  # 1e-6 converts cm³ to m³
    
    # Calculate total width of the depletion region plus intrinsic layer
    denominator = np.sqrt(term1 + term2)
    
    # Calculate total capacitance
    C = (epsilon * A_m2) / denominator
    
    return C



def get_3dB_frequency(R, xi, A, N_A, N_D, n_i, es, V0):
    '''Calculates the 3dB frequency of a photodetector.
    Input:
    R (float): Load resistance (Ohms).
    xi (float): Intrinsic width of the depletion region (μm).
    A (float): Detector Area (μm^2).
    N_A (float): Doping concentration of the p-type region (cm^-3).
    N_D (float): Doping concentration of the n-type region (cm^-3).
    n_i: float, intrinsic carrier density # cm^{-3}
    es (float): Relative permittivity.
    V0 (float): Applied voltage to the PN junction (V).
    Output:
    f_3dB (float): 3dB frequency (Hz).
    '''
    # Calculate the capacitance using the provided capacitance function
    C = capacitance(xi, A, N_A, N_D, n_i, es, V0)
    # Compute 3dB frequency using the formula f_3dB = 1/(2πRC)
    f_3dB = 1.0 / (2 * np.pi * R * C)
    return f_3dB
