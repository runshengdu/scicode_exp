import numpy as np

def find_equilibrium(spins, N, T, J, num_steps):
    '''Find the thermal euilibrium state of a given spin system
    Input:
    spins: starting spin state, 1D array of 1 and -1
    N: size of spin system, int
    T: temprature, float
    J: interaction matrix, 2D array of floats
    num_steps: number of sampling steps per spin in the Monte Carlo simulation, int
    Output:
    spins: final spin state after Monte Carlo simulation, 1D array of 1 and -1
    '''
    # Create a copy to avoid modifying the input spin array
    spins = spins.copy()
    total_steps = N * num_steps
    
    for _ in range(total_steps):
        # Randomly select a spin index to consider flipping
        k = np.random.randint(0, N)
        
        # Calculate the sum of interactions for the selected spin with all other spins
        sum_interactions = np.dot(J[:k, k], spins[:k]) + np.dot(J[k, k+1:], spins[k+1:])
        
        # Compute the energy change if we flip the selected spin
        delta_H = 2 * spins[k] * sum_interactions
        
        # Determine whether to accept the spin flip based on Metropolis criterion
        if delta_H <= 0:
            accept = True
        else:
            if T == 0.0:
                accept = False
            else:
                boltzmann_factor = np.exp(-delta_H / T)
                random_num = np.random.rand()
                accept = random_num < boltzmann_factor
        
        # Flip the spin if the move is accepted
        if accept:
            spins[k] *= -1
    
    return spins


def calculate_overlap(replicas):
    '''Calculate all overlaps in an emsemble of replicas
    Input:
    replicas: list of replicas, list of 1D arrays of 1 and -1
    Output:
    overlaps: pairwise overlap values between all replicas, 1D array of floats, sorted
    '''
    M = len(replicas)
    if M < 2:
        return np.array([])
    
    # Convert list of replicas to a 2D numpy array
    replicas_arr = np.array(replicas)
    N = replicas_arr.shape[1]
    
    # Compute pairwise dot products between all replicas
    dot_products = np.dot(replicas_arr, replicas_arr.T)
    
    # Extract upper triangular part (excluding diagonal) to get unique pairs
    i_indices, j_indices = np.triu_indices(M, k=1)
    overlaps = dot_products[i_indices, j_indices] / N
    
    # Sort overlaps in ascending order
    overlaps_sorted = np.sort(overlaps)
    
    return overlaps_sorted


def analyze_rsb(overlaps, N):
    '''Analyze if the overlap distribution to identify broad peak or multimodal behavior, indicating potential replica symmetry breaking.
    Input:
    overlaps: all overlap values between replicas from all realization, 1D array of floats
    N: size of spin system, int
    Output:
    potential_RSB: True if potential RSB is indicated, False otherwise, boolean
    '''
    M = len(overlaps)
    if M < 1:
        return False
    
    # Check 1: Standard deviation significantly exceeds expected value
    expected_std = 1.0 / np.sqrt(N)
    actual_std = np.std(overlaps)
    std_check = False
    
    if M >= 2:
        # Standard error of sample standard deviation for normal distribution
        se_std = expected_std / np.sqrt(2 * (M - 1))
        # Check if actual std is more than 2 standard errors above expected
        std_check = actual_std > expected_std + 2 * se_std
    
    # Check 2: Multi-modal distribution detection using Kernel Density Estimation
    multi_modal_check = False
    
    if M >= 3:
        if np.all(overlaps == overlaps[0]):
            multi_modal_check = False
        else:
            # Silverman's bandwidth selection
            sigma = actual_std
            h = (4 * sigma**5 / (3 * M)) ** (1/5)
            
            # Create evaluation grid
            x_min = np.min(overlaps) - 2 * h
            x_max = np.max(overlaps) + 2 * h
            x = np.linspace(x_min, x_max, 200)
            
            # Compute KDE with vectorized operations
            x_2d = x.reshape(-1, 1)
            overlaps_2d = overlaps.reshape(1, -1)
            kde = np.sum(np.exp(-(x_2d - overlaps_2d)**2 / (2 * h**2)), axis=1)
            kde /= (M * h * np.sqrt(2 * np.pi))
            
            # Detect local maxima
            n_kde = len(kde)
            is_max = np.zeros(n_kde, dtype=bool)
            
            if n_kde >= 3:
                is_max[1:-1] = (kde[1:-1] > kde[:-2]) & (kde[1:-1] > kde[2:])
            if n_kde >= 2:
                is_max[0] = kde[0] > kde[1]
                is_max[-1] = kde[-1] > kde[-2]
            
            # Count significant peaks (at least 20% of maximum KDE value)
            max_kde = kde.max()
            significant_peaks = is_max & (kde >= 0.2 * max_kde)
            peak_count = significant_peaks.sum()
            multi_modal_check = (peak_count >= 2)
    
    # Determine potential RSB if either check is satisfied
    potential_RSB = std_check or multi_modal_check
    
    return potential_RSB



def spin_glass(N, T, num_steps, num_replicas, num_realizations):
    '''Simulation the SK model using replica method, analyze overlap and identify potential replica symmetry breaking
    Input:
    N: size of spin system, int
    T: temprature, float
    num_steps: number of sampling steps per spin in the Monte Carlo simulation, int
    num_replicas: number of system replicas in one realization, int
    num_realizations: number of realizations to sample different J's, int
    Output:
    potential_RSB: True if potential RSB is indicated, False otherwise, boolean
    mean: mean of the overall overlap distribution, float
    std: standard deviation of the overall overlap distribution, float
    '''
    all_overlaps = []
    
    for _ in range(num_realizations):
        # Generate interaction matrix J with quenched disorder
        J = np.random.randn(N, N) / np.sqrt(N)
        
        # Generate equilibrium states for all replicas in this realization
        replicas = []
        for _ in range(num_replicas):
            # Random initial spin state
            initial_spins = np.random.choice([-1, 1], size=N)
            # Find equilibrium state using Monte Carlo simulation
            eq_spins = find_equilibrium(initial_spins, N, T, J, num_steps)
            replicas.append(eq_spins)
        
        # Calculate pairwise overlaps for this realization
        overlaps = calculate_overlap(replicas)
        all_overlaps.append(overlaps)
    
    # Combine all overlaps from all realizations into a single array
    if all_overlaps:
        all_overlaps = np.concatenate(all_overlaps)
    else:
        all_overlaps = np.array([])
    
    # Compute statistics and analyze RSB
    if all_overlaps.size == 0:
        mean = 0.0
        std = 0.0
        potential_RSB = False
    else:
        mean = float(np.mean(all_overlaps))
        std = float(np.std(all_overlaps))
        potential_RSB = analyze_rsb(all_overlaps, N)
    
    return potential_RSB, mean, std
