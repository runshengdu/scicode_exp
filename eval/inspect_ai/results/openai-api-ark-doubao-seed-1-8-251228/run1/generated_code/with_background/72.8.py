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
    N = lattice.shape[0]
    s_a = lattice[i, j]
    neighbors = neighbor_list((i, j), N)
    sum_sb = sum(lattice[ni, nj] for ni, nj in neighbors)
    energy = -s_a * sum_sb
    return float(energy)


def energy(lattice):
    '''calculate the total energy for all the site (i, j) of the periodic Ising model with dimension (N, N)
    Args: lattice (np.array): shape (N, N), a 2D array +1 and -1
    Return:
        float: energy 
    '''
    N = lattice.shape[0]
    total_sum = 0.0
    for i in range(N):
        for j in range(N):
            s_ij = lattice[i, j]
            neighbors = neighbor_list((i, j), N)
            # Get unique neighbors to avoid counting duplicate directional interactions
            unique_neighbors = list(set(neighbors))
            for ni, nj in unique_neighbors:
                # Count each unique pair exactly once using lexicographical order
                if (i, j) < (ni, nj):
                    total_sum += s_ij * lattice[ni, nj]
    # Total energy is -sum of unique nearest neighbor pair products
    return -total_sum


def magnetization(spins):
    '''total magnetization of the periodic Ising model with dimension (N, N)
    Args: spins (np.array): shape (N, N), a 2D array +1 and -1
    Return:
        float: 
    '''
    total_mag = np.sum(spins)
    return float(total_mag)


def get_flip_probability_magnetization(lattice, i, j, beta):
    '''Calculate spin flip probability and change in total magnetization.
    Args:
        lat (np.array): shape (N, N), 2D lattice of 1 and -1
        i (int): site index along x
        j (int): site index along y
        beta (float): inverse temperature
    Return:
        A (float): acceptance ratio
        dM (int): change in magnetization after the spin flip
    '''
    N = lattice.shape[0]
    s = lattice[i, j]
    # Get list of nearest neighbors using the provided function
    neighbors = neighbor_list((i, j), N)
    # Sum the spins of the neighboring sites
    sum_neighbors = sum(lattice[ni, nj] for ni, nj in neighbors)
    # Calculate total energy difference upon flipping the spin
    delta_H = 2 * s * sum_neighbors
    # Compute acceptance probability
    exponent = np.exp(-beta * delta_H)
    A = min(1.0, exponent)
    # Calculate change in magnetization
    dM = -2 * s
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
            r = np.random.uniform(0.0, 1.0)
            if r < A:
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
    # Initialize random N x N lattice with spins ±1
    lattice = np.random.choice([1, -1], size=(N, N))
    beta = 1.0 / T
    mag2_list = []
    
    # Calculate initial magnetization squared divided by N^4
    initial_m = magnetization(lattice)
    initial_mag2 = (initial_m ** 2) / (N ** 4)
    mag2_list.append(initial_mag2)
    
    for _ in range(nsweeps):
        # Perform one full Metropolis sweep over all spins
        lattice = flip(lattice, beta)
        # Compute current magnetization and its squared value normalized by N^4
        current_m = magnetization(lattice)
        current_mag2 = (current_m ** 2) / (N ** 4)
        mag2_list.append(current_mag2)
    
    # Convert the list to a numpy array for return
    mag2 = np.array(mag2_list)
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
        mag2_array = run(T, N, nsweeps)
        avg_mag2 = np.mean(mag2_array)
        mag2_avg.append(avg_mag2)
    return mag2_avg
