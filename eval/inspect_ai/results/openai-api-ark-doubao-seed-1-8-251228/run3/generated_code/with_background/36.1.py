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
