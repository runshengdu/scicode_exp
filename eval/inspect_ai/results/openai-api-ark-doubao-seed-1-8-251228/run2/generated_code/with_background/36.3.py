import numpy as np
from scipy.integrate import quad
from scipy.optimize import newton


def generation(P, A, lambda_i, alpha, tau, x):
    '''This function computes the excess electron distribution.
    Input:
    P (float): incident optical power in W
    A (float): beam area in μm^2
    lambda_i (float): incident wavelength in nm
    alpha (float): absorption coefficient in cm^-1
    tau (float): lifetime of excess carriers in s
    x (float): depth variable in μm
    Output:
    dN (float): generated carrier density in cm^-3
    '''
    # Physical constants
    h = 6.626e-34  # Planck's constant in J·s
    c = 3e8        # Speed of light in m/s
    
    # Convert wavelength from nm to meters
    lambda_m = lambda_i * 1e-9
    
    # Calculate photon energy in Joules
    hnu = h * c / lambda_m
    
    # Convert beam area from μm² to cm² (1 μm² = 1e-8 cm²)
    A_cm2 = A * 1e-8
    
    # Convert depth x from μm to cm (1 μm = 1e-4 cm)
    x_cm = x * 1e-4
    
    # Compute the exponential absorption term
    exp_term = np.exp(-alpha * x_cm)
    
    # Calculate numerator and denominator for the carrier density formula
    numerator = tau * alpha * P * exp_term
    denominator = A_cm2 * hnu
    
    # Compute excess carrier density
    dN = numerator / denominator
    
    return dN


def fermi_dirac_integral_half_polylog(Ef):
    '''Function to compute the Fermi-Dirac integral of order 1/2 using polylog
    Input:
    Ef (float): Fermi level compared to the conduction band (eV)
    Output:
    n (float): electron density (cm^-3)
    '''
    # Physical constants
    m0 = 9.109e-31   # Electron rest mass in kg
    kBT_eV = 0.0259  # Thermal energy at room temperature in eV
    q = 1.602e-19    # Electron charge in C
    h = 6.626e-34    # Planck's constant in J·s
    
    # Effective electron mass for GaAs
    m_star = 0.067 * m0
    
    # Convert thermal energy from eV to Joules
    kBT_J = kBT_eV * q
    
    # Calculate effective density of states in conduction band (N_c) in m^-3
    term_inside = (2 * np.pi * m_star * kBT_J) / (h ** 2)
    N_c_m3 = 2 * (term_inside) ** (3/2)
    
    # Convert N_c from m^-3 to cm^-3
    N_c = N_c_m3 * 1e-6
    
    # Calculate dimensionless Fermi level
    eta_F = Ef / kBT_eV
    
    # Define integrand for the Fermi-Dirac integral
    integrand = lambda eps: np.sqrt(eps) / (1 + np.exp(eps - eta_F))
    
    # Compute the integral from 0 to infinity
    integral, _ = quad(integrand, 0, np.inf)
    
    # Calculate the Fermi-Dirac integral of order 1/2
    F_half = (2 / np.sqrt(np.pi)) * integral
    
    # Compute electron density
    n = N_c * F_half
    
    return n



def inverse_fermi_dirac_integral_half_polylog_newton(P, A, lambda_i, alpha, tau, x, n=None):
    '''This function uses the Newton-Raphson method to find the root of an implicit function.
    Inputs:
    P (float): incident optical power in W
    A (float): beam area in μm^2
    lambda_i (float): incident wavelength in nm
    alpha (float): absorption coefficient in cm^-1
    tau (float): lifetime of excess carriers in s
    x (float): depth variable in μm
    n (float): electron density, which is unknown at default (set as None)
    Outputs:
    Ef: Fermi level
    '''
    # Determine target carrier density
    if n is None:
        target_n = generation(P, A, lambda_i, alpha, tau, x)
    else:
        if n <= 0:
            raise ValueError("Carrier density n must be a positive number.")
        target_n = n
    
    # Physical constants for calculating effective density of states (N_c)
    m0 = 9.109e-31   # Electron rest mass in kg
    kBT_eV = 0.0259  # Thermal energy at room temperature in eV
    q = 1.602e-19    # Electron charge in C
    h = 6.626e-34    # Planck's constant in J·s
    m_star = 0.067 * m0  # Effective electron mass for GaAs
    kBT_J = kBT_eV * q
    
    # Calculate effective density of states in conduction band (N_c) in cm^-3
    term_inside = (2 * np.pi * m_star * kBT_J) / (h ** 2)
    N_c_m3 = 2 * (term_inside) ** (3/2)
    N_c = N_c_m3 * 1e-6  # Convert from m^-3 to cm^-3
    
    # Initial guess for Fermi level using non-degenerate approximation
    eta_F_guess = np.log(target_n / N_c)
    Ef_guess = eta_F_guess * kBT_eV
    
    # Define root-finding function
    def f(Ef):
        return fermi_dirac_integral_half_polylog(Ef) - target_n
    
    # Solve for Fermi level using Newton-Raphson method
    Ef_solution = newton(f, Ef_guess)
    
    return Ef_solution
