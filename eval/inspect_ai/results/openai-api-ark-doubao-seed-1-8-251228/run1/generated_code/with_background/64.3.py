import numpy as np
import itertools

def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    coord = np.mod(r, L)
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
    r1_np = np.array(r1)
    r2_np = np.array(r2)
    dr = r1_np - r2_np
    # Apply minimum image convention to the difference vector
    dr = np.mod(dr + L / 2, L) - L / 2
    # Calculate Euclidean distance of the adjusted difference vector
    distance = np.linalg.norm(dr)
    return distance



def E_ij(r, sigma, epsilon):
    '''Calculate the Lennard-Jones potential energy between two particles.
    Parameters:
    r : float
        The distance between the two particles.
    sigma : float
        The distance at which the potential minimum occurs
    epsilon : float
        The depth of the potential well
    Returns:
    float
        The potential energy between the two particles at distance r.
    '''
    sr = sigma / r
    sr12 = sr ** 12
    sr6 = sr ** 6
    E_lj = 4 * epsilon * (sr12 - sr6)
    return E_lj
