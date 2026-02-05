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
    delta_x = r1[0] - r2[0]
    delta_y = r1[1] - r2[1]
    delta_z = r1[2] - r2[2]
    
    # Apply minimum image correction to each coordinate difference
    delta_x = delta_x - L * round(delta_x / L)
    delta_y = delta_y - L * round(delta_y / L)
    delta_z = delta_z - L * round(delta_z / L)
    
    # Calculate the Euclidean distance
    distance = math.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
    
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
    lj_r = 4 * epsilon * (sigma_over_r ** 12 - sigma_over_r ** 6)
    
    # Calculate Lennard-Jones potential at cutoff distance rc
    sigma_over_rc = sigma / rc
    lj_rc = 4 * epsilon * (sigma_over_rc ** 12 - sigma_over_rc ** 6)
    
    # Compute truncated and shifted potential
    truncated_shifted_lj = lj_r - lj_rc
    
    return truncated_shifted_lj


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
    num_atoms = xyz.shape[0]
    for i in range(num_atoms):
        r1 = xyz[i]
        for j in range(i + 1, num_atoms):
            r2 = xyz[j]
            distance = dist(r1, r2, L)
            total_energy += E_ij(distance, sigma, epsilon, rc)
    return total_energy



def f_ij(r, sigma, epsilon, rc):
    '''Calculate the force vector between two particles, considering both the truncated and shifted
    Lennard-Jones potential and, optionally, the Yukawa potential.
    Parameters:
    r (float): The distance between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    array_like: The force vector experienced by particle i due to particle j, considering the specified potentials
    '''
    # Convert displacement vector to numpy array for consistent calculations
    r_vec = np.asarray(r)
    r_mag = np.linalg.norm(r_vec)
    
    # Force is zero when distance exceeds cutoff
    if r_mag >= rc:
        return np.zeros_like(r_vec)
    
    # Calculate derivative of Lennard-Jones potential with respect to distance
    sigma_over_r = sigma / r_mag
    dV_dr = 4 * epsilon * (
        -12 * (sigma_over_r ** 12) / r_mag +
        6 * (sigma_over_r ** 6) / r_mag
    )
    
    # Compute force vector using negative gradient of potential
    force = -dV_dr * (r_vec / r_mag)
    
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
    # Initialize force array with zeros matching the shape of position array
    f_xyz = np.zeros_like(xyz)
    
    # Iterate over all unique particle pairs to avoid redundant calculations
    for i in range(N):
        for j in range(i + 1, N):
            # Compute raw displacement vector from particle j to particle i
            delta_vec = xyz[i] - xyz[j]
            
            # Apply minimum image convention to each component of the displacement
            for k in range(3):
                delta_vec[k] -= L * round(delta_vec[k] / L)
            
            # Calculate force exerted on particle i by particle j
            force_ij = f_ij(delta_vec, sigma, epsilon, rc)
            
            # Update net forces using Newton's Third Law (action-reaction pair)
            f_xyz[i] += force_ij
            f_xyz[j] -= force_ij
    
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
    L: float
       length of cubic box
    rc: float
       cutoff distance beyond which the potentials are truncated and shifted to zero
    Outputs:
    new_positions: 2D array of floats with shape (N,3)
                   new positions of all atoms, N is the number of atoms, 3 is x,y,z coordinate
    new_velocities: 2D array of floats with shape (N,3)
              new velocities of all atoms, N is the number of atoms, 3 is x,y,z coordinate
    '''
    # Convert inputs to numpy arrays for consistent operations
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    
    # Calculate current forces and acceleration
    current_forces = forces(N, positions, L, sigma, epsilon, rc)
    current_accel = current_forces / m
    
    # Compute half-step velocity
    v_half = velocities + 0.5 * current_accel * dt
    
    # Update positions
    new_positions = positions + v_half * dt
    
    # Wrap positions into periodic boundary conditions
    new_positions = new_positions % L
    
    # Calculate new forces and acceleration from updated positions
    new_forces = forces(N, new_positions, L, sigma, epsilon, rc)
    new_accel = new_forces / m
    
    # Compute final velocity
    new_velocities = v_half + 0.5 * new_accel * dt
    
    return new_positions, new_velocities
