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
    h = 6.626e-34
    c = 3e8
    
    # Unit conversions to ensure consistency
    x_cm = x * 1e-4  # Convert depth from μm to cm
    A_cm2 = A * 1e-8  # Convert area from μm² to cm²
    lambda_m = lambda_i * 1e-9  # Convert wavelength from nm to m
    
    # Calculate photon energy
    hnu = h * c / lambda_m
    
    # Exponential absorption term
    exp_term = np.exp(-alpha * x_cm)
    
    # Compute excess carrier density using the derived formula
    dN = (tau * alpha * P * exp_term) / (A_cm2 * hnu)
    
    return dN


def fermi_dirac_integral_half_polylog(Ef):
    '''Function to compute the Fermi-Dirac integral of order 1/2 using polylog
    Input:
    Ef (float): Fermi level compared to the conduction band (eV)
    Output:
    n (float): electron density (cm^-3)
    '''
    # Physical constants
    m0 = 9.109e-31  # Rest electron mass in kg
    m_star = 0.067 * m0  # Effective electron mass in kg
    q = 1.602e-19  # Electron charge in C
    kBT_eV = 0.0259  # Thermal voltage in eV at room temperature
    kBT_J = kBT_eV * q  # Thermal energy in J
    h = 6.626e-34  # Planck's constant in J·s
    
    # Calculate effective density of states in conduction band (N_c)
    term = (2 * np.pi * m_star * kBT_J) / (h ** 2)
    N_c_SI = 2 * (term) ** (3/2)  # N_c in m^-3
    N_c = N_c_SI * 1e-6  # Convert to cm^-3
    
    # Calculate dimensionless Fermi level
    eta_F = Ef / kBT_eV
    
    # Compute the Fermi integral using numerical integration
    integral, _ = quad(
        lambda eps: np.sqrt(eps) / (1 + np.exp(eps - eta_F)),
        0, np.inf
    )
    
    # Calculate Fermi-Dirac integral of order 1/2
    fermi_integral = (2 / np.sqrt(np.pi)) * integral
    
    # Compute electron density
    n = N_c * fermi_integral
    
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
    # Calculate electron density using generation function if not provided
    if n is None:
        n = generation(P, A, lambda_i, alpha, tau, x)
    
    # Physical constants for effective density of states calculation
    m0 = 9.109e-31
    m_star = 0.067 * m0
    q = 1.602e-19
    kBT_eV = 0.0259
    kBT_J = kBT_eV * q
    h = 6.626e-34
    
    # Compute effective density of states in conduction band (N_c)
    term = (2 * np.pi * m_star * kBT_J) / (h ** 2)
    N_c_SI = 2 * (term) ** (3/2)
    N_c = N_c_SI * 1e-6  # Convert to cm^-3
    
    # Initial guess using non-degenerate approximation
    if n <= 0:
        initial_guess = 0.0
    else:
        initial_guess = kBT_eV * np.log(n / N_c)
    
    # Define the function to find root of: f(Ef) = n_calculated - n_target
    def f(Ef):
        return fermi_dirac_integral_half_polylog(Ef) - n
    
    # Use Newton-Raphson method to find the Fermi level
    Ef = newton(f, initial_guess)
    
    return Ef
