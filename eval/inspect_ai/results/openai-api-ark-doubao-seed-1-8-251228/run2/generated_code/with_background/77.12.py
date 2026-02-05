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


def forces(N, xyz, L, sigma, epsilon, rc):
    '''Calculate the net forces acting on each particle in a system due to all pairwise interactions.
    Parameters:
    N : int
        The number of particles in the system.
    xyz : ndarray
        A NumPy array with shape (N, 3) containing the positions of each particle in the system,
        in nanometers.
    L : float
        The length of the side of the cubic simulation box (in nanometers), used for applying the minimum
        image convention in periodic boundary conditions.
    sigma : float
        The Lennard-Jones size parameter (in nanometers), indicating the distance at which the
        inter-particle potential is zero.
    epsilon : float
        The depth of the potential well (in zeptojoules), indicating the strength of the particle interactions.
    rc : float
        The cutoff distance (in nanometers) beyond which the inter-particle forces are considered negligible.
    Returns:
    ndarray
        A NumPy array of shape (N, 3) containing the net force vectors acting on each particle in the system,
        in zeptojoules per nanometer (zJ/nm).
    '''
    xyz_np = np.asarray(xyz, dtype=np.float64)
    f_xyz = np.zeros_like(xyz_np, dtype=np.float64)
    
    for i in range(N):
        r_i = xyz_np[i]
        for j in range(i + 1, N):
            r_j = xyz_np[j]
            delta = r_i - r_j
            delta_mic = np.mod(delta + L / 2, L) - L / 2
            f_ij_vec = f_ij(delta_mic, sigma, epsilon, rc)
            f_xyz[i] += f_ij_vec
            f_xyz[j] -= f_ij_vec
    
    return f_xyz



def velocityVerlet(N, xyz, v_xyz, L, sigma, epsilon, rc, m, dt, tau_T, T_target, tau_P, P_target):
    '''Integrate the equations of motion using the velocity Verlet algorithm, with the inclusion of the Berendsen thermostat
    and barostat for temperature and pressure control, respectively.
    Parameters:
    N : int
        The number of particles in the system.
    xyz : ndarray
        Current particle positions in the system, shape (N, 3), units: nanometers.
    v_xyz : ndarray
        Current particle velocities in the system, shape (N, 3), units: nanometers/ps.
    L : float
        Length of the cubic simulation box's side, units: nanometers.
    sigma : float
        Lennard-Jones potential size parameter, units: nanometers.
    epsilon : float
        Lennard-Jones potential depth parameter, units: zeptojoules.
    rc : float
        Cutoff radius for potential calculation, units: nanometers.
    m : float
        Mass of each particle, units: grams/mole.
    dt : float
        Integration timestep, units: picoseconds.
    tau_T : float
        Temperature coupling time constant for the Berendsen thermostat. Set to 0 to deactivate, units: picoseconds.
    T_target : float
        Target temperature for the Berendsen thermostat, units: Kelvin.
    tau_P : float
        Pressure coupling time constant for the Berendsen barostat. Set to 0 to deactivate, units: picoseconds.
    P_target : float
        Target pressure for the Berendsen barostat, units: bar.
    Returns:
    --------
    xyz_full : ndarray
        Updated particle positions in the system, shape (N, 3), units: nanometers.
    v_xyz_full : ndarray
        Updated particle velocities in the system, shape (N, 3), units: nanometers/ps.
    L : float
        Updated length of the cubic simulation box's side, units: nanometers.
    Raises:
    -------
    Exception:
        If the Berendsen barostat has shrunk the box such that the side length L is less than twice the cutoff radius.
    Exception:
        If the Berendsen barostat has resulted in non-positive box length.
    '''
    xyz_np = np.asarray(xyz, dtype=np.float64)
    v_xyz_np = np.asarray(v_xyz, dtype=np.float64)
    
    # Step 1: Calculate initial interatomic forces and acceleration
    f_interatomic = forces(N, xyz_np, L, sigma, epsilon, rc)
    mass_factor = Avogadro / (m * 1e24)  # Convert force (zJ/nm) to acceleration (nm/ps²)
    a_interatomic = f_interatomic * mass_factor
    
    # Step 2: Calculate instantaneous temperature for thermostat
    T_current = temperature(v_xyz_np, m, N)
    
    # Step 3: Apply Berendsen thermostat acceleration term if active
    if tau_T > 0:
        delta_T = (T_target / T_current) - 1
        a_berendsen_t = (1 / (2 * tau_T)) * delta_T * v_xyz_np
    else:
        a_berendsen_t = np.zeros_like(v_xyz_np)
    a_t = a_interatomic + a_berendsen_t
    
    # Step 4: Calculate half-step velocity
    v_half = v_xyz_np + 0.5 * a_t * dt
    
    # Step 5: Calculate tentative new positions
    x_temp = xyz_np + v_half * dt
    # Wrap positions to original box bounds
    x_temp_wrapped = wrap(x_temp, L)
    
    # Step 6: Apply Berendsen barostat if active
    if tau_P > 0:
        # Calculate current system pressure
        _, _, P_current = pressure(N, L, T_current, xyz_np, sigma, epsilon, rc)
        # Isothermal compressibility (typical value for liquid argon in bar⁻¹)
        gamma = 1e-4
        # Calculate scaling factor
        eta = 1 - (dt / tau_P) * gamma * (P_target - P_current)
        # Update box length
        L_new = L * eta
        
        # Validate new box length
        if L_new <= 0:
            raise Exception("Berendsen barostat has resulted in non-positive box length.")
        if L_new < 2 * rc:
            raise Exception("Berendsen barostat has shrunk the box such that the side length L is less than twice the cutoff radius.")
        
        # Scale positions to new box dimensions
        x_scaled = x_temp_wrapped * eta
    else:
        L_new = L
        x_scaled = x_temp_wrapped
    
    # Step 7: Calculate new interatomic forces and acceleration
    f_interatomic_new = forces(N, x_scaled, L_new, sigma, epsilon, rc)
    a_interatomic_new = f_interatomic_new * mass_factor
    
    # Step 8: Apply Berendsen thermostat acceleration term for final step
    if tau_T > 0:
        delta_T = (T_target / T_current) - 1
        a_berendsen_new = (1 / (2 * tau_T)) * delta_T * v_half
    else:
        a_berendsen_new = np.zeros_like(v_half)
    a_t_plus_dt = a_interatomic_new + a_berendsen_new
    
    # Step 9: Calculate final velocity
    v_xyz_full = v_half + 0.5 * a_t_plus_dt * dt
    
    return x_scaled, v_xyz_full, L_new
