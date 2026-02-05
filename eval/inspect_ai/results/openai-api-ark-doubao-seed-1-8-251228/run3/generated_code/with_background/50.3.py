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
    
    N = len(replicas[0])
    # Generate all unique pairwise overlaps
    overlaps_list = [np.dot(replicas[i], replicas[j]) / N for i in range(M) for j in range(i + 1, M)]
    
    # Sort overlaps in ascending order
    overlaps_list.sort()
    
    # Convert to numpy array
    return np.array(overlaps_list)



def analyze_rsb(overlaps, N):
    '''Analyze if the overlap distribution to identify broad peak or multimodal behavior, indicating potential replica symmetry breaking.
    Input:
    overlaps: all overlap values between replicas from all realization, 1D array of floats
    N: size of spin system, int
    Output:
    potential_RSB: True if potential RSB is indicated, False otherwise, boolean
    '''
    n = len(overlaps)
    if n < 2:
        return False  # Insufficient data to make a judgment
    
    mu = np.mean(overlaps)
    sigma = np.std(overlaps, ddof=1)
    expected_std = 1.0 / np.sqrt(N)
    
    # Check if mean is significantly different from 0
    sem = sigma / np.sqrt(n)
    if sem == 0:
        # All overlaps are identical
        mean_significant = not np.isclose(mu, 0)
    else:
        z_score = abs(mu) / sem
        mean_significant = z_score > 2  # 95% confidence threshold
    
    # Check if distribution is broader than expected
    std_significant = sigma > 1.5 * expected_std
    
    # Check for multimodal behavior using Kernel Density Estimation
    has_multiple_peaks = False
    if n >= 20 and sigma > 0:
        # Silverman's rule for bandwidth selection
        h = 1.06 * sigma * (n) ** (-1/5)
        min_val = np.min(overlaps)
        max_val = np.max(overlaps)
        
        # Generate evaluation points for KDE
        x = np.linspace(min_val - 2 * h, max_val + 2 * h, 1000)
        
        # Compute Kernel Density Estimate
        exponent = -(x[:, np.newaxis] - overlaps) ** 2 / (2 * h ** 2)
        kde = np.sum(np.exp(exponent), axis=1) / (n * h)
        
        # Detect number of peaks in KDE
        if len(kde) >= 2:
            diffs = np.diff(kde)
            signs = np.sign(diffs)
            sign_changes = np.diff(signs)
            
            # Count sign changes from positive to negative (indicating peaks)
            peak_indices = np.where(sign_changes == -2)[0] + 1
            peaks = len(peak_indices)
            
            # Check edge points for peaks
            if kde[0] > kde[1]:
                peaks += 1
            if kde[-1] > kde[-2]:
                peaks += 1
            
            has_multiple_peaks = peaks >= 2
    
    # Determine potential RSB if any condition is met
    potential_RSB = mean_significant or std_significant or has_multiple_peaks
    
    return potential_RSB
