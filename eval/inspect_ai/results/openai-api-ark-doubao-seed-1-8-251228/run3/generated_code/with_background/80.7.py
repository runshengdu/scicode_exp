import os
import math
import time
import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
import pickle
from scipy.constants import  Avogadro

def dist(r1, r2, L):
    '''Calculate the minimum image distance between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    float: The minimum image distance between the two atoms.
    '''
    # Compute the coordinate differences
    dx = r1[0] - r2[0]
    dy = r1[1] - r2[1]
    dz = r1[2] - r2[2]
    
    # Apply minimum image correction to each component
    dx = dx - L * round(dx / L)
    dy = dy - L * round(dy / L)
    dz = dz - L * round(dz / L)
    
    # Calculate Euclidean distance
    distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    
    return distance


def E_ij(r, sigma, epsilon, rc):
    '''Calculate the combined truncated and shifted Lennard-Jones potential energy and,
    if specified, the truncated and shifted Yukawa potential energy between two particles.
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
    
    # Calculate Lennard-Jones potential at distance r
    sigma_over_r = sigma / r
    v_lj_r = 4 * epsilon * (sigma_over_r ** 12 - sigma_over_r ** 6)
    
    # Calculate Lennard-Jones potential at cutoff distance rc
    sigma_over_rc = sigma / rc
    v_lj_rc = 4 * epsilon * (sigma_over_rc ** 12 - sigma_over_rc ** 6)
    
    # Return truncated and shifted potential
    return v_lj_r - v_lj_rc


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
    total_energy = 0.0
    num_particles = xyz.shape[0]
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            r_i = xyz[i]
            r_j = xyz[j]
            r = dist(r_i, r_j, L)
            total_energy += E_ij(r, sigma, epsilon, rc)
    return total_energy


def f_ij(r, sigma, epsilon, rc):
    '''Calculate the force vector between two particles, considering both the truncated and shifted
    Lennard-Jones potential and, optionally, the Yukawa potential.
    Parameters:
    r (array_like): The 3D displacement vector between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    array_like: The force vector experienced by particle i due to particle j, considering the specified potentials
    '''
    r = np.asarray(r)
    r_mag = np.linalg.norm(r)
    
    if r_mag >= rc or r_mag == 0.0:
        return np.zeros(3)
    
    sigma_over_r = sigma / r_mag
    dV_dr = 24 * epsilon / r_mag * (sigma_over_r ** 6 - 2 * sigma_over_r ** 12)
    force = -dV_dr * (r / r_mag)
    
    return force


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
    # Initialize force array with zeros matching the shape of xyz
    f_xyz = np.zeros_like(xyz)
    
    # Iterate over all unique particle pairs (i < j) to avoid double counting
    for i in range(N):
        for j in range(i + 1, N):
            # Calculate raw displacement vector between particle i and j
            dr = xyz[i] - xyz[j]
            # Apply minimum image correction for periodic boundary conditions
            dr -= L * np.round(dr / L)
            # Compute force on particle i due to particle j
            fij = f_ij(dr, sigma, epsilon, rc)
            # Update net forces using Newton's third law
            f_xyz[i] += fij
            f_xyz[j] -= fij
    
    return f_xyz


def velocity_verlet(N, sigma, epsilon, positions, velocities, dt, m, L, rc):
    '''This function runs Velocity Verlet algorithm to integrate the positions and velocities of atoms interacting through
    Lennard Jones Potential forward for one time step according to Newton's Second Law.
    Inputs:
    N: int
       The number of particles in the system.
    sigma: float
       the distance at which Lennard Jones potential reaches zero
    epsilon: float
             potential well depth of Lennard Jones potential
    positions: 2D array of floats with shape (N,3)
              current positions of all atoms, N is the number of atoms, 3 is x,y,z coordinate
    velocities: 2D array of floats with shape (N,3)
              current velocities of all atoms, N is the number of atoms, 3 is x,y,z coordinate
    dt: float
        time step size
    m: float
       mass
    Outputs:
    new_positions: 2D array of floats with shape (N,3)
                   new positions of all atoms, N is the number of atoms, 3 is x,y,z coordinate
    new_velocities: 2D array of floats with shape (N,3)
              current velocities of all atoms, N is the number of atoms, 3 is x,y,z coordinate
    '''
    # Compute current acceleration from net forces (a = F/m)
    current_forces = forces(N, positions, L, sigma, epsilon, rc)
    a_t = current_forces / m
    
    # Calculate half-step velocity
    v_half = velocities + 0.5 * a_t * dt
    
    # Update positions and apply periodic boundary conditions
    new_positions = positions + v_half * dt
    new_positions = np.mod(new_positions, L)
    
    # Compute new acceleration from forces at updated positions
    new_forces = forces(N, new_positions, L, sigma, epsilon, rc)
    a_t_plus_dt = new_forces / m
    
    # Calculate final velocity at full time step
    new_velocities = v_half + 0.5 * a_t_plus_dt * dt
    
    return new_positions, new_velocities



def MD_NVT(N, init_positions, init_velocities, L, sigma, epsilon, rc, m, dt, num_steps, T, nu):
    '''Integrate the equations of motion using the velocity Verlet algorithm, with the inclusion of the Berendsen thermostat
    and barostat for temperature and pressure control, respectively.
    Parameters:
    N: int
       The number of particles in the system.
    init_positions: 2D array of floats with shape (N,3)
              current positions of all atoms, N is the number of atoms, 3 is x,y,z coordinate, units: nanometers.
    init_velocities: 2D array of floats with shape (N,3)
              current velocities of all atoms, N is the number of atoms, 3 is x,y,z coordinate, units: nanometers.
    L: float
        Length of the cubic simulation box's side, units: nanometers.
    sigma: float
       the distance at which Lennard Jones potential reaches zero, units: nanometers.
    epsilon: float
             potential well depth of Lennard Jones potential, units: zeptojoules.
    rc: float
        Cutoff radius for potential calculation, units: nanometers.
    m: float
       Mass of each particle, units: grams/mole.
    dt: float
        Integration timestep, units: picoseconds
    num_steps: float
        step number
    T: float
      Current temperature of the particles
    nu: float
      Frequency of the collosion
    Returns:
    tuple
        Updated positions and velocities of all particles, and the possibly modified box length after barostat adjustment.
    '''
    # Convert num_steps to integer
    num_steps = int(num_steps)
    
    # Initialize positions and velocities
    positions = np.copy(init_positions)
    velocities = np.copy(init_velocities)
    
    # Initialize arrays to store total energy and instant temperature
    E_total_array = np.zeros(num_steps)
    instant_T_array = np.zeros(num_steps)
    
    # Initialize intercollision times tracking
    intercollision_times = []
    last_collision_time = np.zeros(N)  # Tracks last collision time for each particle, starts at 0
    
    # Compute mass per particle in kg once (m is g/mol)
    m_p_kg = (m / 1000.0) / sp.constants.Avogadro
    
    # Compute standard deviation for Maxwell-Boltzmann velocity components (nm/ps)
    if T > 0:
        kBT = sp.constants.Boltzmann * T  # Boltzmann constant * temperature in Joules
        std_dev_SI = np.sqrt(kBT / m_p_kg)  # Standard deviation in m/s
        std_dev = std_dev_SI * 1e-3  # Convert to nm/ps (1 m/s = 1e-3 nm/ps)
    else:
        # At T=0, velocities are zero
        std_dev = 0.0
    
    # Probability of a particle colliding during one time step
    p = 1.0 - np.exp(-nu * dt) if nu > 0 else 0.0
    
    for step in range(num_steps):
        # Calculate potential energy using E_pot function
        potential_energy = E_pot(positions, L, sigma, epsilon, rc)
        
        # Calculate kinetic energy
        sum_v_sq = np.sum(velocities ** 2)  # Sum of (nm/ps)^2 for all velocity components
        sum_v_sq_SI = sum_v_sq * 1e6  # Convert (nm/ps)^2 to (m/s)^2 (1 nm/ps = 1e3 m/s)
        KE_J = 0.5 * m_p_kg * sum_v_sq_SI
        KE_zJ = KE_J * 1e21  # Convert Joules to zeptojoules (1 J = 1e21 zJ)
        
        # Total energy
        total_energy = potential_energy + KE_zJ
        E_total_array[step] = total_energy
        
        # Calculate instant temperature
        if N == 0 or KE_J == 0:
            instant_T = 0.0
        else:
            instant_T = (2 * KE_J) / (3 * N * sp.constants.Boltzmann)
        instant_T_array[step] = instant_T
        
        # Perform Velocity Verlet step to get new positions and velocities
        new_positions, new_velocities = velocity_verlet(N, sigma, epsilon, positions, velocities, dt, m, L, rc)
        
        # Apply Anderson thermostat
        for i in range(N):
            if nu > 0 and np.random.rand() < p:
                # Sample new velocity from Maxwell-Boltzmann distribution
                new_vel = np.random.normal(0.0, std_dev, size=3)
                new_velocities[i] = new_vel
                # Record intercollision time
                current_time = (step + 1) * dt
                inter_coll_time = current_time - last_collision_time[i]
                intercollision_times.append(inter_coll_time)
                last_collision_time[i] = current_time
        
        # Update positions and velocities for next iteration
        positions = new_positions
        velocities = new_velocities
    
    # Convert intercollision_times to numpy array for consistency
    intercollision_times = np.array(intercollision_times)
    
    return (E_total_array, instant_T_array, positions, velocities, intercollision_times)
