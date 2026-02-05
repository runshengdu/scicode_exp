import numpy as np



def KL_divergence(p, q):
    '''Input
    p: probability distributions, 1-dimensional numpy array (or list) of floats
    q: probability distributions, 1-dimensional numpy array (or list) of floats
    Output
    divergence: KL-divergence of two probability distributions, a single scalar value (float)
    '''
    p_np = np.asarray(p)
    q_np = np.asarray(q)
    
    # Mask to avoid terms where p is zero (since 0 * log(...) = 0)
    non_zero_mask = p_np > 0
    p_non_zero = p_np[non_zero_mask]
    q_non_zero = q_np[non_zero_mask]
    
    # Compute each non-zero term and sum
    terms = p_non_zero * np.log2(p_non_zero / q_non_zero)
    divergence = np.sum(terms)
    
    return divergence
