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


def run(T, N, nsweeps):
    '''Performs Metropolis to flip spins for nsweeps times and collect iteration, temperature, energy, and magnetization^2 in a dataframe
    Args: 
        T (float): temperature
        N (int): system size along an axis
        nsweeps: number of iterations to go over all spins
    Return:
        mag2: (numpy array) magnetization^2
    '''
    # Initialize random lattice of 1 and -1
    lattice = np.random.choice([1, -1], size=(N, N))
    beta = 1.0 / T
    # Initialize array to store magnetization squared values
    mag2 = np.zeros(nsweeps)
    
    for sweep in range(nsweeps):
        # Perform one Metropolis sweep
        lattice = flip(lattice, beta)
        # Calculate total magnetization
        m = magnetization(lattice)
        # Store magnetization squared
        mag2[sweep] = m ** 2
    
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
    N4 = N ** 4  # Precompute N^4 once for efficiency
    for T in Ts:
        # Get magnetization squared values for all sweeps at current temperature
        mag2_array = run(T, N, nsweeps)
        # Calculate average magnetization squared over all sweeps
        avg_m2 = np.mean(mag2_array)
        # Compute magnetization^2 / N^4 and add to the result list
        mag2_over_N4 = avg_m2 / N4
        mag2_avg.append(mag2_over_N4)
    return mag2_avg



def calc_transition(T_list, mag2_list):
    '''Calculates the transition temperature by taking derivative
    Args:
        T_list: list of temperatures
        mag2_list: list of magnetization^2/N^4 at each temperature
    Return:
        float: Transition temperature
    '''
    # Convert input lists to numpy arrays for numerical operations
    T_array = np.array(T_list)
    mag2_array = np.array(mag2_list)
    
    # Compute numerical derivative of magnetization squared with respect to temperature
    d_mag2_dT = np.gradient(mag2_array, T_array)
    
    # Find index of the minimum derivative value
    min_deriv_idx = np.argmin(d_mag2_dT)
    
    # Get the corresponding transition temperature
    T_transition = T_array[min_deriv_idx]
    
    return float(T_transition)
