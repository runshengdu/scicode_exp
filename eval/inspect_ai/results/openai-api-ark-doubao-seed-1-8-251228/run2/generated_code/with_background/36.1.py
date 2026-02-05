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
