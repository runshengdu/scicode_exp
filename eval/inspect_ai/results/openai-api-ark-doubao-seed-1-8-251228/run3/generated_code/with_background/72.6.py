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
    if N == 1:
        return 0.0
    total = 0.0
    for i in range(N):
        for j in range(N):
            total += energy_site(i, j, lattice)
    if N == 2:
        return total / 4
    else:
        return total / 2


def magnetization(spins):
    '''total magnetization of the periodic Ising model with dimension (N, N)
    Args: spins (np.array): shape (N, N), a 2D array +1 and -1
    Return:
        float: 
    '''
    total_spin = spins.sum()
    return float(total_spin)


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
    dM = -2 * s
    
    E = energy_site(i, j, lattice)
    delta_H = -2 * E
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
    lattice = spins.copy()
    N = lattice.shape[0]
    for i in range(N):
        for j in range(N):
            A, _ = get_flip_probability_magnetization(lattice, i, j, beta)
            u = np.random.rand()
            if u < A:
                lattice[i, j] *= -1
    return lattice
