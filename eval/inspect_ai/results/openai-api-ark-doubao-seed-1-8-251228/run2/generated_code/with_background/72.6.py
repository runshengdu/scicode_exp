import numpy as np

def neighbor_list(site, N):
    '''Return all nearest neighbors of site (i, j).
    Args:
        site (Tuple[int, int]): site indices
        N (int): number of sites along each dimension
    Return:
        list: a list of 2-tuples, [(i_left, j_left), (i_above, j_above), (i_right, j_right), (i_below, j_below)]
    '''
    i, j = site
    # Calculate each neighbor with periodic boundary conditions using modulo
    left = (i, (j - 1) % N)
    above = ((i - 1) % N, j)
    right = (i, (j + 1) % N)
    below = ((i + 1) % N, j)
    nn_wrap = [left, above, right, below]
    return nn_wrap


def energy_site(i, j, lattice):
    '''Calculate the energy of site (i, j)
    Args:
        i (int): site index along x
        j (int): site index along y
        lattice (np.array): shape (N, N), a 2D array +1 and -1
    Return:
        float: energy of site (i, j)
    '''
    s_ij = lattice[i, j]
    N = lattice.shape[0]
    neighbors = neighbor_list((i, j), N)
    sum_neighbors = sum(lattice[ni, nj] for ni, nj in neighbors)
    energy = -s_ij * sum_neighbors
    return float(energy)


def energy(lattice):
    '''calculate the total energy for the site (i, j) of the periodic Ising model with dimension (N, N)
    Args: lattice (np.array): shape (N, N), a 2D array +1 and -1
    Return:
        float: energy 
    '''
    N = lattice.shape[0]
    total_energy = 0.0
    for i in range(N):
        for j in range(N):
            total_energy += energy_site(i, j, lattice)
    # Each nearest-neighbor pair is counted twice in the sum of site energies, so divide by 2
    return total_energy / 2.0


def magnetization(spins):
    '''total magnetization of the periodic Ising model with dimension (N, N)
    Args: spins (np.array): shape (N, N), a 2D array +1 and -1
    Return:
        float: 
    '''
    return float(spins.sum())


def get_flip_probability_magnetization(lattice, i, j, beta):
    '''Calculate spin flip probability and change in total magnetization.
    Args:
        lattice (np.array): shape (N, N), 2D lattice of 1 and -1
        i (int): site index along x
        j (int): site index along y
        beta (float): inverse temperature
    Return:
        A (float): acceptance ratio
        dM (int): change in magnetization after the spin flip
    '''
    s = lattice[i, j]
    # Calculate change in magnetization
    dM = -2 * s
    # Calculate original site energy
    E_ij = energy_site(i, j, lattice)
    # Calculate total energy difference after flipping the spin
    delta_H = -2 * E_ij
    # Compute acceptance probability
    exponent = -beta * delta_H
    A = min(1.0, np.exp(exponent))
    return A, dM



def flip(spins, beta):
    '''Goes through each spin in the 2D lattice and flip it.
    Args:
        spins (np.array): shape (N, N), 2D lattice of 1 and -1        
        beta (float): inverse temperature
    Return:
        lattice (np.array): final spin configurations
    '''
    # Create a copy of the input lattice to avoid modifying the original
    lattice = spins.copy()
    N = lattice.shape[0]
    
    # Iterate over each spin site in the lattice
    for i in range(N):
        for j in range(N):
            # Calculate acceptance probability for flipping the current site
            acceptance_prob, _ = get_flip_probability_magnetization(lattice, i, j, beta)
            # Generate uniform random number between 0 and 1
            random_num = np.random.rand()
            # Accept the flip if random number is less than acceptance probability
            if random_num < acceptance_prob:
                lattice[i, j] *= -1
    
    return lattice
