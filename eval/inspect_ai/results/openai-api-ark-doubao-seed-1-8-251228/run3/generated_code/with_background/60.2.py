import numpy as np

def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    r_array = np.asarray(r)
    coord = np.mod(r_array, L)
    return coord



def E_i(r, pos, sigma, epsilon, L, r_c):
    '''Calculate the total Lennard-Jones potential energy of a particle with other particles in a periodic system.
    Parameters:
    r : array, the (x, y, z) coordinates of the target particle.
    pos : An array of (x, y, z) coordinates for each of the other particles in the system.
    sigma : float, the distance at which the potential minimum occurs
    epsilon : float, the depth of the potential well
    L : float, the length of the side of the cubic box
    r_c : float, cut-off distance
    Returns:
    float, the total Lennard-Jones potential energy of the particle due to its interactions with other particles.
    '''
    def E_ij(r_ij):
        '''Compute the Lennard-Jones Potential between a pair of atoms.'''
        if r_ij < r_c:
            sigma_over_r = sigma / r_ij
            return 4 * epsilon * (sigma_over_r ** 12 - sigma_over_r ** 6)
        return 0.0
    
    total_energy = 0.0
    r_array = np.asarray(r)
    pos_array = np.asarray(pos)
    
    for pos_j in pos_array:
        # Calculate separation vector between target particle and current particle
        delta = pos_j - r_array
        # Apply minimum image convention to get shortest periodic distance
        delta = delta - L * np.round(delta / L)
        # Compute Euclidean distance between the particles
        r_ij = np.linalg.norm(delta)
        # Accumulate pairwise potential energy
        total_energy += E_ij(r_ij)
    
    return total_energy
