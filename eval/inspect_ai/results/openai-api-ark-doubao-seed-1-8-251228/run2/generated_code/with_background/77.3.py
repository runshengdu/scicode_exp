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
