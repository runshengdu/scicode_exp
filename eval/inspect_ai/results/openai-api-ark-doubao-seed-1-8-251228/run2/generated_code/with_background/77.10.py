import math
import numpy as np
import scipy as sp
from scipy.constants import  Avogadro

def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    r_np = np.asarray(r, dtype=np.float64)
    coord = np.mod(r_np, L)
    return coord


def dist(r1, r2, L):
    '''Calculate the minimum image distance between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    float: The minimum image distance between the two atoms.
    '''
    r1_np = np.asarray(r1, dtype=np.float64)
    r2_np = np.asarray(r2, dtype=np.float64)
    delta = r1_np - r2_np
    # Apply minimum image convention to each component of the difference vector
    delta_mic = np.mod(delta + L / 2, L) - L / 2
    # Compute Euclidean norm of the corrected difference vector
    distance = np.linalg.norm(delta_mic)
    return distance


def dist_v(r1, r2, L):
    '''Calculate the minimum image vector between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    numpy 1d array of floats: The minimum image vector between the two atoms.
    '''
    r1_np = np.asarray(r1, dtype=np.float64)
    r2_np = np.asarray(r2, dtype=np.float64)
    delta = r1_np - r2_np
    # Apply minimum image convention to each component of the difference vector
    delta_mic = np.mod(delta + L / 2, L) - L / 2
    return delta_mic


def E_ij(r, sigma, epsilon, rc):
    '''Calculate the combined truncated and shifted Lennard-Jones potential energy between two particles.
    Parameters:
    r (float): The distance between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    float: The combined potential energy between the two particles, considering the specified potentials.
    '''
    if r >= rc:
        return 0.0
    
    # Compute Lennard-Jones potential at distance r
    sigma_over_r = sigma / r
    term6 = sigma_over_r ** 6
    term12 = term6 ** 2
    v_lj_r = 4 * epsilon * (term12 - term6)
    
    # Compute Lennard-Jones potential at cutoff distance rc
    sigma_over_rc = sigma / rc
    term6_c = sigma_over_rc ** 6
    term12_c = term6_c ** 2
    v_lj_rc = 4 * epsilon * (term12_c - term6_c)
    
    # Calculate truncated and shifted potential
    E = v_lj_r - v_lj_rc
    return E


def f_ij(r, sigma, epsilon, rc):
    '''Calculate the force vector between two particles, considering the truncated and shifted
    Lennard-Jones potential.
    Parameters:
    r (float): The distance between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    array_like: The force vector experienced by particle i due to particle j, considering the specified potentials
    '''
    r_np = np.asarray(r, dtype=np.float64)
    r_scalar = np.linalg.norm(r_np)
    
    if r_scalar >= rc:
        return np.zeros_like(r_np)
    
    sigma_over_r = sigma / r_scalar
    sigma_over_r6 = sigma_over_r ** 6
    sigma_over_r12 = sigma_over_r6 ** 2
    
    dV_dr = 4 * epsilon * (-12 * sigma_over_r12 / r_scalar + 6 * sigma_over_r6 / r_scalar)
    force_vector = -dV_dr * (r_np / r_scalar)
    
    return force_vector


def E_tail(N, L, sigma, epsilon, rc):
    '''Calculate the energy tail correction for a system of particles, considering the truncated and shifted
    Lennard-Jones potential.
    Parameters:
    N (int): The total number of particles in the system.
    L (float): Length of cubic box
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    float
        The energy tail correction for the entire system (in zeptojoules), considering the specified potentials.
    '''
    volume = L ** 3
    sigma_over_rc = sigma / rc
    term9 = sigma_over_rc ** 9
    term3 = sigma_over_rc ** 3
    bracket = (1/3) * term9 - term3
    pre_factor = (8 * math.pi * N ** 2 * epsilon * sigma ** 3) / (3 * volume)
    E_tail_LJ = pre_factor * bracket
    return E_tail_LJ


def P_tail(N, L, sigma, epsilon, rc):
    ''' Calculate the pressure tail correction for a system of particles, including
     the truncated and shifted Lennard-Jones contributions.
    Parameters:
     N (int): The total number of particles in the system.
     L (float): Lenght of cubic box
     sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
     epsilon (float): The depth of the potential well for the Lennard-Jones potential.
     rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
     Returns:
     float
         The pressure tail correction for the entire system (in bar).
     
    '''
    volume = L ** 3
    sigma_over_rc = sigma / rc
    term9 = sigma_over_rc ** 9
    term3 = sigma_over_rc ** 3
    bracket = (2 / 3) * term9 - term3
    
    pre_factor = (16 / 3) * math.pi * (N ** 2) * epsilon * (sigma ** 3)
    p_tail_zj_per_nm3 = pre_factor * bracket / (volume ** 2)
    
    # Convert from zJ/nm³ to bar (1 zJ/nm³ = 10 bar)
    p_tail_bar = p_tail_zj_per_nm3 * 10
    
    return p_tail_bar


def E_pot(xyz, L, sigma, epsilon, rc):
    '''Calculate the total potential energy of a system using the truncated and shifted Lennard-Jones potential.
    Parameters:
    xyz : A NumPy array with shape (N, 3) where N is the number of particles. Each row contains the x, y, z coordinates of a particle in the system.
    L (float): Length of cubic box
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    float
        The total potential energy of the system (in zeptojoules).
    '''
    xyz_np = np.asarray(xyz, dtype=np.float64)
    N = xyz_np.shape[0]
    total_energy = 0.0
    
    for i in range(N):
        for j in range(i + 1, N):
            r1 = xyz_np[i]
            r2 = xyz_np[j]
            distance = dist(r1, r2, L)
            pair_energy = E_ij(distance, sigma, epsilon, rc)
            total_energy += pair_energy
    
    return total_energy


def temperature(v_xyz, m, N):
    '''Calculate the instantaneous temperature of a system of particles using the equipartition theorem.
    Parameters:
    v_xyz : ndarray
        A NumPy array with shape (N, 3) containing the velocities of each particle in the system,
        in nanometers per picosecond (nm/ps).
    m : float
        The molar mass of the particles in the system, in grams per mole (g/mol).
    N : int
        The number of particles in the system.
    Returns:
    float
        The instantaneous temperature of the system in Kelvin (K).
    '''
    v_xyz_np = np.asarray(v_xyz, dtype=np.float64)
    sum_v_sq = np.sum(v_xyz_np ** 2)
    k_B = 0.0138064852  # Boltzmann constant in zJ/K
    # Calculate total kinetic energy in zJ
    ke_total = 0.5 * (m / Avogadro) * sum_v_sq * 1e24
    # Solve for temperature using equipartition theorem
    T = (2 * ke_total) / (3 * N * k_B)
    return T



def pressure(N, L, T, xyz, sigma, epsilon, rc):
    '''Calculate the pressure of a system of particles using the virial theorem, considering
    the Lennard-Jones contributions.
    Parameters:
    N : int
        The number of particles in the system.
    L : float
        The length of the side of the cubic simulation box (in nanometers).
    T : float
        The instantaneous temperature of the system (in Kelvin).
    xyz : ndarray
        A NumPy array with shape (N, 3) containing the positions of each particle in the system, in nanometers.
    sigma : float
        The Lennard-Jones size parameter (in nanometers).
    epsilon : float
        The depth of the potential well (in zeptojoules).
    rc : float
        The cutoff distance beyond which the inter-particle potential is considered to be zero (in nanometers).
    Returns:
    tuple
        The kinetic pressure (in bar), the virial pressure (in bar), and the total pressure (kinetic plus virial, in bar) of the system.
    '''


    k_B = 0.0138064852  # Boltzmann constant in zJ/K
    V = L ** 3

    # Calculate kinetic pressure
    P_kinetic_zj_nm3 = (N * k_B * T) / V
    P_kinetic = P_kinetic_zj_nm3 * 10  # Convert zJ/nm³ to bar (1 zJ/nm³ = 10 bar)

    # Calculate virial pressure
    xyz_np = np.asarray(xyz, dtype=np.float64)
    virial_sum = 0.0

    for i in range(N):
        r_i = xyz_np[i]
        for j in range(i + 1, N):
            r_j = xyz_np[j]
            # Compute minimum image displacement from j to i (r_i - r_j wrapped to periodic boundaries)
            delta = r_i - r_j
            delta_mic = np.mod(delta + L / 2, L) - L / 2
            # Get force on i due to j using f_ij function
            f_ij = f_ij(delta_mic, sigma, epsilon, rc)
            # Displacement from i to j as per problem statement
            r_ij = r_j - r_i
            # Calculate dot product and add to virial sum
            virial_sum += np.dot(f_ij, r_ij)

    P_virial_zj_nm3 = virial_sum / (3 * V)
    P_virial = P_virial_zj_nm3 * 10  # Convert zJ/nm³ to bar

    # Calculate total pressure
    P_total = P_kinetic + P_virial

    return P_kinetic, P_virial, P_total
