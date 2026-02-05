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
