import numpy as np



def KL_divergence(p, q):
    '''Input
    p: probability distributions, 1-dimensional numpy array (or list) of floats
    q: probability distributions, 1-dimensional numpy array (or list) of floats
    Output
    divergence: KL-divergence of two probability distributions, a single scalar value (float)
    '''
    p_arr = np.asarray(p)
    q_arr = np.asarray(q)
    
    # Mask to select non-zero elements in p, as terms with p(x)=0 contribute 0
    non_zero_mask = p_arr > 0
    p_non_zero = p_arr[non_zero_mask]
    q_non_zero = q_arr[non_zero_mask]
    
    # Calculate KL terms for non-zero p elements
    kl_terms = p_non_zero * np.log2(p_non_zero / q_non_zero)
    divergence = np.sum(kl_terms)
    
    return divergence
