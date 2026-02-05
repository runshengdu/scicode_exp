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
    # Calculate the coordinate differences
    dr = np.array(r1) - np.array(r2)
    # Apply minimum image convention to each component
    dr = dr - L * np.round(dr / L)
    # Compute Euclidean distance of the adjusted difference vector
    distance = np.linalg.norm(dr)
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
    
    # Return truncated and shifted potential
    return lj_r - lj_rc



def E_pot(xyz, L, sigma, epsilon, rc):
    '''Calculate the total potential energy of a system using the truncated and shifted Lennard-Jones potential.
    Parameters:
    xyz : A NumPy array with shape (N, 3) where N is the number of particles. Each row contains the x, y, z coordinates of a particle in the system.
    L (float): Lenght of cubic box
    r (float): The distance between particles i and j.
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
            # Calculate minimum image distance between particles i and j
            r = dist(xyz[i], xyz[j], L)
            # Add pairwise Lennard-Jones potential energy to total
            total_energy += E_ij(r, sigma, epsilon, rc)
    return total_energy
