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
