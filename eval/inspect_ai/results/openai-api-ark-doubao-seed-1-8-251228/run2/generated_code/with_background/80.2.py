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
