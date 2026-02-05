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


def run(T, N, nsweeps):
    '''Performs Metropolis to flip spins for nsweeps times and collect iteration, temperature, energy, and magnetization^2 in a dataframe
    Args: 
        T (float): temperature
        N (int): system size along an axis
        nsweeps: number of iterations to go over all spins
    Return:
        mag2: (numpy array) magnetization^2
    '''
    lattice = np.random.choice([1, -1], size=(N, N))
    beta = 1.0 / T
    mag2 = np.zeros(nsweeps, dtype=float)
    
    for sweep in range(nsweeps):
        lattice = flip(lattice, beta)
        m = magnetization(lattice)
        mag2[sweep] = (m ** 2) / (N ** 4)
    
    return mag2



def scan_T(Ts, N, nsweeps):
    '''Run over several given temperatures.
    Args:
        Ts: list of temperature
        N: system size in one axis
        nsweeps: number of iterations to go over all spins
    Return:
        mag2_avg: list of magnetization^2 / N^4, each element is the value for each temperature
    '''
    mag2_avg = []
    for T in Ts:
        mag2 = run(T, N, nsweeps)
        avg_mag2 = np.mean(mag2)
        mag2_avg.append(avg_mag2)
    return mag2_avg
