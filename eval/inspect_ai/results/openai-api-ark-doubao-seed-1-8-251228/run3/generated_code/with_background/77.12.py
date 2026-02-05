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
    r_np = np.asarray(r)
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
    r1_np = np.asarray(r1)
    r2_np = np.asarray(r2)
    dr = r1_np - r2_np
    # Adjust displacement to minimum image convention
    dr_adjusted = np.mod(dr + L / 2, L) - L / 2
    # Compute Euclidean distance
    distance = np.linalg.norm(dr_adjusted)
    return distance


def dist_v(r1, r2, L):
    '''Calculate the minimum image vector between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    numpy.ndarray: The minimum image vector between the two atoms.
    '''
    r1_np = np.asarray(r1)
    r2_np = np.asarray(r2)
    dr = r1_np - r2_np
    # Adjust each component to follow minimum image convention
    dr_adjusted = np.mod(dr + L / 2, L) - L / 2
    return dr_adjusted


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
    sigma_r = sigma / r
    term_r = (sigma_r ** 12) - (sigma_r ** 6)
    sigma_rc = sigma / rc
    term_rc = (sigma_rc ** 12) - (sigma_rc ** 6)
    return 4 * epsilon * (term_r - term_rc)


def f_ij(r, sigma, epsilon, rc):
    '''Calculate the force vector between two particles, considering the truncated and shifted
    Lennard-Jones potential.
    Parameters:
    r (array_like): The displacement vector (r1 - r2) between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    array_like: The force vector experienced by particle i due to particle j, considering the specified potentials
    '''
    r_np = np.asarray(r)
    r_scalar = np.linalg.norm(r_np)
    
    if r_scalar >= rc:
        return np.zeros_like(r_np)
    
    # Calculate terms for the force vector
    sigma6 = sigma ** 6
    sigma12 = sigma ** 12
    r8 = r_scalar ** 8
    r14 = r_scalar ** 14
    
    term1 = 48 * epsilon * sigma12 / r14
    term2 = 24 * epsilon * sigma6 / r8
    
    force_vector = (term1 - term2) * r_np
    return force_vector


def E_tail(N, L, sigma, epsilon, rc):
    '''Calculate the energy tail correction for a system of particles, considering the truncated and shifted
    Lennard-Jones potential.
    Parameters:
    N (int): The total number of particles in the system.
    L (float): Lenght of cubic box
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    float
        The energy tail correction for the entire system (in zeptojoules), considering the specified potentials.
    '''
    volume = L ** 3
    sigma_over_rc = sigma / rc
    bracket_term = (1/3) * (sigma_over_rc) ** 9 - sigma_over_rc ** 3
    E_tail_LJ = (8 * math.pi * N ** 2 * epsilon * sigma ** 3) / (3 * volume) * bracket_term
    return E_tail_LJ


def P_tail(N, L, sigma, epsilon, rc):
    ''' Calculate the pressure tail correction for a system of particles, including
     the truncated and shifted Lennard-Jones contributions.
    Parameters:
     N (int): The total number of particles in the system.
     L (float): Length of cubic box
     sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
     epsilon (float): The depth of the potential well for the Lennard-Jones potential.
     rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
     Returns:
     float
         The pressure tail correction for the entire system (in bar).
     
    '''
    volume = L ** 3
    sigma_over_rc = sigma / rc
    bracket_term = (2/3) * (sigma_over_rc) ** 9 - (sigma_over_rc) ** 3
    # Calculate pressure in Pascals
    p_pa = (16 * math.pi * N ** 2 * epsilon * sigma ** 3) / (3 * volume ** 2) * bracket_term
    # Convert Pascals to bar (1 bar = 100000 Pa)
    P_tail_bar = p_pa / 1e5
    return P_tail_bar


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
    xyz_np = np.asarray(xyz)
    total_energy = 0.0
    N = xyz_np.shape[0]
    for i in range(N):
        r1 = xyz_np[i]
        for j in range(i + 1, N):
            r2 = xyz_np[j]
            distance = dist(r1, r2, L)
            total_energy += E_ij(distance, sigma, epsilon, rc)
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
    k_B_J = 0.0138064852e-21  # Boltzmann constant in J/K
    v_xyz_np = np.asarray(v_xyz)
    # Calculate sum of squared velocities over all particle components
    sum_v_sq = np.sum(v_xyz_np ** 2)
    # Calculate sum of m_p * v² for all particles in Joules
    sum_m_p_v_sq = (m * sum_v_sq * 1000) / Avogadro
    # Compute temperature using equipartition theorem
    T = sum_m_p_v_sq / (3 * N * k_B_J)
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
    # Calculate volume in cubic meters
    V_m3 = (L * 1e-9) ** 3
    k_B = 0.0138064852  # Boltzmann constant in zJ/K
    
    # Compute kinetic pressure
    # N*k_B*T is in zJ, convert to J by multiplying by 1e-21
    P_kinetic_pa = (N * k_B * T * 1e-21) / V_m3
    P_kinetic_bar = P_kinetic_pa / 1e5  # Convert Pa to bar
    
    # Calculate virial pressure
    sum_virial = 0.0
    for i in range(N):
        r1 = xyz[i]
        for j in range(i + 1, N):
            r2 = xyz[j]
            dr = r1 - r2
            # Adjust displacement to minimum image convention
            dr_adjusted = np.mod(dr + L / 2, L) - L / 2
            # Compute force vector using pre-defined f_ij function
            f_ij_vec = f_ij(dr_adjusted, sigma, epsilon, rc)
            # Compute dot product of force and adjusted displacement
            dot_product = np.dot(f_ij_vec, dr_adjusted)
            sum_virial += dot_product
    
    # Convert sum_virial from zJ to J
    sum_virial_J = sum_virial * 1e-21
    # Compute virial pressure in Pascals
    P_virial_pa = sum_virial_J / (3 * V_m3)
    # Convert to bar
    P_virial_bar = P_virial_pa / 1e5
    
    # Calculate total pressure
    P_total_bar = P_kinetic_bar + P_virial_bar
    
    return P_kinetic_bar, P_virial_bar, P_total_bar

# Assume the pre-defined f_ij function is available as provided earlier
def f_ij(r, sigma, epsilon, rc):
    '''Calculate the force vector between two particles, considering the truncated and shifted
    Lennard-Jones potential.
    Parameters:
    r (array_like): The displacement vector (r1 - r2) between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    array_like: The force vector experienced by particle i due to particle j, considering the specified potentials
    '''
    r_np = np.asarray(r)
    r_scalar = np.linalg.norm(r_np)
    
    if r_scalar >= rc:
        return np.zeros_like(r_np)
    
    sigma6 = sigma ** 6
    sigma12 = sigma ** 12
    r8 = r_scalar ** 8
    r14 = r_scalar ** 14
    
    term1 = 48 * epsilon * sigma12 / r14
    term2 = 24 * epsilon * sigma6 / r8
    
    force_vector = (term1 - term2) * r_np
    return force_vector


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
    # Initialize force array with zeros
    f_xyz = np.zeros((N, 3), dtype=np.float64)
    
    for i in range(N):
        r1 = xyz[i]
        for j in range(i + 1, N):
            r2 = xyz[j]
            # Compute displacement vector from particle j to particle i
            dr = r1 - r2
            # Adjust displacement to follow minimum image convention
            dr_adjusted = np.mod(dr + L / 2, L) - L / 2
            # Calculate force on particle i due to particle j
            force_ij = f_ij(dr_adjusted, sigma, epsilon, rc)
            # Apply forces using Newton's third law (action-reaction pair)
            f_xyz[i] += force_ij
            f_xyz[j] -= force_ij
    
    return f_xyz

# Required dependency function (as provided in problem statement)
def f_ij(r, sigma, epsilon, rc):
    '''Calculate the force vector between two particles, considering the truncated and shifted
    Lennard-Jones potential.
    Parameters:
    r (array_like): The displacement vector (r1 - r2) between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    array_like: The force vector experienced by particle i due to particle j, considering the specified potentials
    '''
    r_np = np.asarray(r)
    r_scalar = np.linalg.norm(r_np)
    
    if r_scalar >= rc:
        return np.zeros_like(r_np)
    
    sigma6 = sigma ** 6
    sigma12 = sigma ** 12
    r8 = r_scalar ** 8
    r14 = r_scalar ** 14
    
    term1 = 48 * epsilon * sigma12 / r14
    term2 = 24 * epsilon * sigma6 / r8
    
    force_vector = (term1 - term2) * r_np
    return force_vector



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
    '''
    # Helper function to compute acceleration from forces
    def get_acceleration(forces, m):
        # Convert force from zJ/nm to N: 1 zJ/nm = 1e-12 N
        forces_N = forces * 1e-12
        # Mass per particle in kg: m (g/mol) -> kg/particle
        m_p_kg = (m * 1e-3) / Avogadro
        # Acceleration in m/s²
        accel_m_s2 = forces_N / m_p_kg
        # Convert to nm/(ps²): 1 m/s² = 1e-15 nm/(ps²)
        accel_nm_ps2 = accel_m_s2 * 1e-15
        return accel_nm_ps2
    
    # Step 1: Compute initial forces and acceleration
    f_initial = forces(N, xyz, L, sigma, epsilon, rc)
    a_initial = get_acceleration(f_initial, m)
    
    # Step 2: Apply Berendsen thermostat (add acceleration term)
    if tau_T > 0:
        # Compute current temperature
        T_current = temperature(v_xyz, m, N)
        # Thermostat acceleration
        a_thermo = (1.0 / (2.0 * tau_T)) * (T_target / T_current - 1.0) * v_xyz
        a_total = a_initial + a_thermo
    else:
        a_total = a_initial
    
    # Step 3: Compute half-step velocity
    v_half = v_xyz + 0.5 * a_total * dt
    
    # Step 4: Compute new positions (unscaled for barostat)
    x_unscaled = xyz + v_half * dt
    # Wrap positions to periodic boundaries
    x_unscaled_wrapped = np.array([wrap(r, L) for r in x_unscaled])
    
    # Step 5: Apply Berendsen barostat if active
    if tau_P > 0:
        # Compute current pressure (total pressure)
        T_current = temperature(v_xyz, m, N)
        P_kinetic, P_virial, P_current = pressure(N, L, T_current, xyz, sigma, epsilon, rc)
        
        # Isothermal compressibility for Lennard-Jones fluid (gamma = 1.0e-4 bar⁻¹, standard value)
        gamma = 1.0e-4  # bar⁻¹
        
        # Calculate scaling factor eta
        eta = 1.0 - (dt / tau_P) * gamma * (P_target - P_current)
        
        # Scale positions and box length
        x_scaled = x_unscaled_wrapped * eta
        L_new = L * eta
        
        # Check if box is too small
        if L_new < 2 * rc:
            raise Exception(f"Box length {L_new} nm is less than twice cutoff radius {2*rc} nm")
        
        # Update positions and box length
        x_new = x_scaled
        L = L_new
        
        # Scale half-step velocity to maintain momentum (since volume changed)
        v_half_scaled = v_half * eta
    else:
        x_new = x_unscaled_wrapped
        v_half_scaled = v_half
    
    # Step 6: Compute new forces and acceleration from scaled positions
    f_new = forces(N, x_new, L, sigma, epsilon, rc)
    a_new = get_acceleration(f_new, m)
    
    # Step 7: Apply Berendsen thermostat to new acceleration if active
    if tau_T > 0:
        # Compute temperature from half-step velocity (best estimate for T(t+dt/2))
        # Create dummy velocity array for temperature calculation
        T_half = temperature(v_half_scaled, m, N)
        a_thermo_new = (1.0 / (2.0 * tau_T)) * (T_target / T_half - 1.0) * v_half_scaled
        a_total_new = a_new + a_thermo_new
    else:
        a_total_new = a_new
    
    # Step 8: Compute full-step velocity
    v_full = v_half_scaled + 0.5 * a_total_new * dt
    
    # Return updated positions, velocities, and box length
    return x_new, v_full, L

# Required dependency functions (as defined in previous steps)
def forces(N, xyz, L, sigma, epsilon, rc):
    f_xyz = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        r1 = xyz[i]
        for j in range(i + 1, N):
            r2 = xyz[j]
            dr = r1 - r2
            dr_adjusted = np.mod(dr + L / 2, L) - L / 2
            force_ij = f_ij(dr_adjusted, sigma, epsilon, rc)
            f_xyz[i] += force_ij
            f_xyz[j] -= force_ij
    return f_xyz

def f_ij(r, sigma, epsilon, rc):
    r_np = np.asarray(r)
    r_scalar = np.linalg.norm(r_np)
    if r_scalar >= rc:
        return np.zeros_like(r_np)
    sigma6 = sigma ** 6
    sigma12 = sigma ** 12
    r8 = r_scalar ** 8
    r14 = r_scalar ** 14
    term1 = 48 * epsilon * sigma12 / r14
    term2 = 24 * epsilon * sigma6 / r8
    force_vector = (term1 - term2) * r_np
    return force_vector

def temperature(v_xyz, m, N):
    k_B_J = 0.0138064852e-21  # Boltzmann constant in J/K
    v_xyz_np = np.asarray(v_xyz)
    sum_v_sq = np.sum(v_xyz_np ** 2)
    sum_m_p_v_sq = (m * sum_v_sq * 1000) / Avogadro
    T = sum_m_p_v_sq / (3 * N * k_B_J)
    return T

def pressure(N, L, T, xyz, sigma, epsilon, rc):
    V_m3 = (L * 1e-9) ** 3
    k_B = 0.0138064852  # Boltzmann constant in zJ/K
    P_kinetic_pa = (N * k_B * T * 1e-21) / V_m3
    P_kinetic_bar = P_kinetic_pa / 1e5
    
    sum_virial = 0.0
    for i in range(N):
        r1 = xyz[i]
        for j in range(i + 1, N):
            r2 = xyz[j]
            dr = r1 - r2
            dr_adjusted = np.mod(dr + L / 2, L) - L / 2
            f_ij_vec = f_ij(dr_adjusted, sigma, epsilon, rc)
            dot_product = np.dot(f_ij_vec, dr_adjusted)
            sum_virial += dot_product
    
    sum_virial_J = sum_virial * 1e-21
    P_virial_pa = sum_virial_J / (3 * V_m3)
    P_virial_bar = P_virial_pa / 1e5
    P_total_bar = P_kinetic_bar + P_virial_bar
    
    return P_kinetic_bar, P_virial_bar, P_total_bar

def wrap(r, L):
    r_np = np.asarray(r)
    coord = np.mod(r_np, L)
    return coord
