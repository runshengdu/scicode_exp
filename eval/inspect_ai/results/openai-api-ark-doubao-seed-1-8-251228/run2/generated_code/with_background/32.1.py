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
    # Calculate wave number
    k = 2 * np.pi / l
    kr = k * R

    # Calculate polarizability
    alpha = 4 * np.pi * epsilon_0 * (a ** 3) * (n ** 2 - 1) / (n ** 2 + 2)

    # Calculate electric field magnitudes
    E1 = np.sqrt(4 * P[0] / (np.pi * w ** 2 * epsilon_0 * c))
    E2 = np.sqrt(4 * P[1] / (np.pi * w ** 2 * epsilon_0 * c))

    # Calculate F_xx components
    coeff_xx = (2 * alpha ** 2 * E1 * E2 * np.cos(phi) ** 2) / (8 * np.pi * epsilon_0 * R ** 4)
    term_xx = -3 * np.cos(kr) - 3 * kr * np.sin(kr) + (kr) ** 2 * np.cos(kr)
    F_xx = coeff_xx * term_xx

    # Calculate F_xy components
    coeff_xy = (alpha ** 2 * E1 * E2 * np.sin(phi) ** 2) / (8 * np.pi * epsilon_0 * R ** 4)
    term_xy = 3 * np.cos(kr) + 3 * kr * np.sin(kr) - 2 * (kr) ** 2 * np.cos(kr) - (kr) ** 3 * np.sin(kr)
    F_xy = coeff_xy * term_xy

    # Total binding force
    F = F_xx + F_xy

    return F
