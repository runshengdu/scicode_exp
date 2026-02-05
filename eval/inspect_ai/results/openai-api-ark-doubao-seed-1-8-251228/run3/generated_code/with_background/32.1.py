import numpy as np
import scipy
from scipy.constants import epsilon_0, c



def binding_force(P, phi, R, l, w, a, n):
    '''Function to calculate the optical binding force between two trapped nanospheres.
    Input
    P : list of length 2
        Power of the two optical traps.
    phi : float
        Polarization direction of the optical traps.
    R : float
        Distance between the trapped nanospheres.
    l : float
        Wavelength of the optical traps.
    w : float
        Beam waist of the optical traps.
    a : float
        Radius of the trapped microspheres.
    n : float
        Refractive index of the trapped microspheres.
    Output
    F : float
        The optical binding force between two trapped nanospheres.
    '''
    # Calculate polarizability alpha
    alpha = 4 * np.pi * epsilon_0 * (a ** 3) * (n ** 2 - 1) / (n ** 2 + 2)
    
    # Calculate electric field magnitudes for each trap
    E1 = np.sqrt(4 * P[0] / (np.pi * w ** 2 * epsilon_0 * c))
    E2 = np.sqrt(4 * P[1] / (np.pi * w ** 2 * epsilon_0 * c))
    
    # Wave number and scaled distance
    k = 2 * np.pi / l
    kR = k * R
    
    # Precompute common trigonometric and power terms
    cos_kR = np.cos(kR)
    sin_kR = np.sin(kR)
    kR_sq = kR ** 2
    kR_cu = kR ** 3
    
    # Compute F_xx component
    coeff_xx = (2 * alpha ** 2 * E1 * E2 * np.cos(phi) ** 2) / (8 * np.pi * epsilon_0 * R ** 4)
    bracket_xx = -3 * cos_kR - 3 * kR * sin_kR + kR_sq * cos_kR
    F_xx = coeff_xx * bracket_xx
    
    # Compute F_xy component
    coeff_xy = (alpha ** 2 * E1 * E2 * np.sin(phi) ** 2) / (8 * np.pi * epsilon_0 * R ** 4)
    bracket_xy = 3 * cos_kR + 3 * kR * sin_kR - 2 * kR_sq * cos_kR - kR_cu * sin_kR
    F_xy = coeff_xy * bracket_xy
    
    # Total optical binding force
    F = F_xx + F_xy
    
    return F
