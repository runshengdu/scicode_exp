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
    # Convert coordinates to numpy arrays for element-wise operations
    r1_np = np.asarray(r1)
    r2_np = np.asarray(r2)
    
    # Compute the difference vector between the two atoms
    dr = r1_np - r2_np
    
    # Apply minimum image correction to each component of the difference vector
    # This adjusts each component to the range [-L/2, L/2]
    dr -= L * np.round(dr / L)
    
    # Calculate the Euclidean norm of the corrected difference vector
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
    E_lj = 4 * epsilon * (sr ** 12 - sr ** 6)
    return E_lj



def E_i(r, positions, L, sigma, epsilon):
    '''Calculate the total Lennard-Jones potential energy of a particle with other particles in a periodic system.
    Parameters:
    r : array_like
        The (x, y, z) coordinates of the target particle.
    positions : array_like
        An array of (x, y, z) coordinates for each of the other particles in the system.
    L : float
        The length of the side of the cubic box
    sigma : float
        The distance at which the potential minimum occurs
    epsilon : float
        The depth of the potential well
    Returns:
    float
        The total Lennard-Jones potential energy of the particle due to its interactions with other particles.
    '''
    total_energy = 0.0
    for pos in positions:
        distance = dist(r, pos, L)
        total_energy += E_ij(distance, sigma, epsilon)
    return total_energy
