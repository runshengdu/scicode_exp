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
    # Calculate polarizability alpha
    alpha = 4 * np.pi * epsilon_0 * (a ** 3) * (n ** 2 - 1) / (n ** 2 + 2)
    
    # Calculate mass of each nanosphere (density * volume)
    m = (4 / 3) * np.pi * (a ** 3) * rho
    
    # Calculate trap spring constants k_i for each sphere
    k_list = []
    for pi in P:
        E = np.sqrt(4 * pi / (np.pi * w ** 2 * epsilon_0 * c))
        ki = 2 * alpha * (E ** 2) / (w ** 2)
        k_list.append(ki)
    k_list = np.array(k_list)
    
    # Initialize k_ij matrix (coupling from binding force derivatives)
    k_ij_mat = np.zeros((N, N), dtype=np.float64)
    
    # Compute k_ij for all i != j using central difference numerical differentiation
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # Equilibrium separation between sphere i and j
            D = abs(i - j) * R
            # Compute binding force at separation D + h and D - h
            F_plus = binding_force([P[i], P[j]], phi, D + h, l, w, a, n)
            F_minus = binding_force([P[i], P[j]], phi, D - h, l, w, a, n)
            # Calculate derivative (central difference)
            k_ij = (F_plus - F_minus) / (2 * h)
            k_ij_mat[i, j] = k_ij
    
    # Sum of k_ij for each sphere i (over all j != i)
    sum_k = np.sum(k_ij_mat, axis=1)
    
    # Calculate resonant frequencies Omega_i
    Omega = np.sqrt((k_list + sum_k) / m)
    
    # Initialize Hamiltonian matrix
    H = np.zeros((N, N), dtype=np.float64)
    
    # Fill diagonal elements with resonant frequencies
    for i in range(N):
        H[i, i] = Omega[i]
    
    # Fill off-diagonal elements with coupling constants g_ij
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            denominator = 2 * m * np.sqrt(Omega[i] * Omega[j])
            g_ij = -k_ij_mat[i, j] / denominator
            H[i, j] = g_ij
    
    return H



def runge_kutta(C0, H, L, M, t0, steps):
    '''Function to numerically solve the Lindblad master equation with the Runge-Kutta method.
    Input
    C0 : matrix of shape(N, N)
        Initial correlation matrix.
    H : matrix of shape(N, N)
        The Hamiltonian of the system.
    L : matrix of shape(N, N)
        The dissipation matrix.
    M : matrix of shape(N, N)
        The reservoir matrix.
    t0 : float
        The time point at which to calculate the phonon occupation.
    steps : int
        Number of simulation steps for the integration.
    Output
    nf : list of length N
        Phonon occupation of each trapped microsphere at time point `t0`.
    '''
    N = C0.shape[0]
    # Initialize current correlation matrix with complex dtype to handle imaginary components
    C_current = np.asarray(C0, dtype=np.complex128)
    delta_t = t0 / steps if steps != 0 else 0.0

    # Helper function to compute the right-hand side of the master equation
    def rhs(C):
        commutator = H @ C - C @ H
        anticommutator = L @ C + C @ L
        return 1j * commutator + anticommutator + M

    # Perform Runge-Kutta 4th order iterations
    for _ in range(steps):
        k1 = rhs(C_current)
        C_k2 = C_current + k1 * delta_t / 2
        k2 = rhs(C_k2)
        C_k3 = C_current + k2 * delta_t / 2
        k3 = rhs(C_k3)
        C_k4 = C_current + k3 * delta_t
        k4 = rhs(C_k4)
        
        # Update current correlation matrix
        C_current += (delta_t / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Extract real-valued phonon occupations from diagonal elements
    nf = [np.real(C_current[i, i]) for i in range(N)]
    return nf
