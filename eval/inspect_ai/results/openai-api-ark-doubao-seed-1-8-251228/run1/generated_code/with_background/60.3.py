import numpy as np

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
        '''Compute the Lennard-Jones Potential between a pair of atoms at distance r_ij.'''
        if r_ij < r_c:
            sigma_over_r = sigma / r_ij
            return 4 * epsilon * (sigma_over_r ** 12 - sigma_over_r ** 6)
        else:
            return 0.0
    
    r_np = np.asarray(r)
    pos_np = np.asarray(pos)
    total_energy = 0.0
    
    for pos_j in pos_np:
        # Compute relative position vector
        dr = pos_j - r_np
        # Apply minimum image convention to get shortest periodic distance
        dr = np.mod(dr + L / 2, L) - L / 2
        # Calculate Euclidean distance between particles
        r_ij = np.linalg.norm(dr)
        # Accumulate pairwise interaction energy
        total_energy += E_ij(r_ij)
    
    return total_energy



def Widom_insertion(pos, sigma, epsilon, L, r_c, T):
    '''Perform the Widom test particle insertion method to calculate the change in chemical potential.
    Parameters:
    pos : ndarray, Array of position vectors of all particles in the system.
    sigma: float, The effective particle diameter 
    epsilon: float, The depth of the potential well
    L: float, The length of each side of the cubic simulation box
    r_c: float, Cut-Off Distance
    T: float, The temperature of the system
    Returns:
    float: Boltzmann factor for the test particle insertion, e^(-beta * energy of insertion).
    '''
    # Generate random position for the test particle within the cubic box [0, L)
    test_r = np.random.uniform(0.0, L, size=3)
    
    # Calculate the total insertion energy using the E_i function
    U_insert = E_i(test_r, pos, sigma, epsilon, L, r_c)
    
    # Define beta as 1/(k_B*T), assuming k_B=1 (common in reduced unit simulations)
    beta = 1.0 / T
    
    # Compute the Boltzmann factor
    Boltz = np.exp(-beta * U_insert)
    
    return Boltz
