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
    total_steps = num_steps * N
    for _ in range(total_steps):
        # Select random spin index
        k = np.random.randint(0, N)
        
        # Calculate sum of interactions for j < k (using column k)
        sum_j_lt_k = np.dot(J[:k, k], spins[:k])
        # Calculate sum of interactions for j > k (using row k)
        sum_j_gt_k = np.dot(J[k, k+1:], spins[k+1:])
        total_interaction = sum_j_lt_k + sum_j_gt_k
        
        # Compute energy change if we flip spin k
        delta_H = 2 * spins[k] * total_interaction
        
        # Metropolis acceptance criterion
        if delta_H <= 0:
            # Always accept favorable energy change
            spins[k] = -spins[k]
        else:
            # Compute acceptance probability for unfavorable change
            if T > 0:
                acceptance_prob = np.exp(-delta_H / T)
                # Generate random number for acceptance check
                if np.random.rand() < acceptance_prob:
                    spins[k] = -spins[k]
            # At T=0, never accept unfavorable changes
    return spins
