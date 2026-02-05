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



def generate_Hamiltonian(P, phi, R, l, w, a, n, h, N, rho):
    '''Function to generate the Hamiltonian of trapped nanospheres with optical binding force appeared.
    Input
    P : list of length N
        Power of each individual optical trap.
    phi : float
        Polarization direction of the optical traps.
    R : float
        Distance between the adjacent trapped nanospheres.
    l : float
        Wavelength of the optical traps.
    w : float
        Beam waist of the optical traps.
    a : float
        Radius of the trapped microspheres.
    n : float
        Refractive index of the trapped microspheres.
    h : float
        Step size of the differentiation.
    N : int
        The total number of trapped nanospheres.
    rho: float
        Density of the trapped microspheres.
    Output
    H : matrix of shape(N, N)
        The Hamiltonian of trapped nanospheres with optical binding force appeared.
    '''
    # Compute polarizability of the nanospheres
    alpha = 4 * np.pi * epsilon_0 * (a ** 3) * (n ** 2 - 1) / (n ** 2 + 2)
    
    # Compute mass of a single nanosphere (density * volume)
    m = rho * (4/3) * np.pi * (a ** 3)
    
    # Compute trap stiffness for each optical trap
    E = np.sqrt(4 * np.array(P) / (np.pi * w ** 2 * epsilon_0 * c))
    k = 2 * alpha * (E ** 2) / (w ** 2)
    
    # Initialize matrix for coupling constants k_ij
    k_ij = np.zeros((N, N))
    
    # Calculate k_ij for all i != j using central difference differentiation
    for i in range(N):
        for j in range(i + 1, N):
            D = abs(i - j) * R
            # Compute binding force at D + h and D - h
            F_plus = binding_force([P[i], P[j]], phi, D + h, l, w, a, n)
            F_minus = binding_force([P[i], P[j]], phi, D - h, l, w, a, n)
            # Central difference derivative of binding force w.r.t distance
            dF_dD = (F_plus - F_minus) / (2 * h)
            k_ij[i, j] = dF_dD
            k_ij[j, i] = dF_dD
    
    # Calculate sum of coupling constants for each particle
    sum_kij = np.sum(k_ij, axis=1)
    
    # Calculate resonant frequencies of each coupled oscillator
    Omega = np.sqrt((k + sum_kij) / m)
    
    # Build Hamiltonian matrix
    denominator = 2 * m * np.sqrt(Omega[:, np.newaxis] * Omega[np.newaxis, :])
    H = -k_ij / denominator
    # Fill diagonal elements with resonant frequencies
    np.fill_diagonal(H, Omega)
    
    return H
