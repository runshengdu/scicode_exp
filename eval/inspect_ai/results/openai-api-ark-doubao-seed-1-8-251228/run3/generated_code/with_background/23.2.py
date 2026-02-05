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



def mutual_info(channel, prior):
    '''Input
    channel: a classical channel, 2d array of floats; channel[i][j] means probability of i given j
    prior:   input random variable, 1d array of floats.
    Output
    mutual: mutual information between the input random variable and the random variable associated with the output of the channel, a single scalar value (float)
    '''
    prior_arr = np.asarray(prior)
    channel_arr = np.asarray(channel)
    
    # Compute H(X) - Entropy of input distribution
    non_zero_x = prior_arr > 0
    h_x_terms = prior_arr[non_zero_x] * np.log2(prior_arr[non_zero_x])
    h_x = -np.sum(h_x_terms)
    
    # Compute joint distribution p(X,Y) = p(X) * p(Y|X)
    joint = channel_arr * prior_arr
    
    # Compute H(XY) - Joint entropy
    non_zero_joint = joint > 0
    h_xy_terms = joint[non_zero_joint] * np.log2(joint[non_zero_joint])
    h_xy = -np.sum(h_xy_terms)
    
    # Compute output distribution p(Y) = sum_X p(X,Y)
    p_y = channel_arr @ prior_arr
    
    # Compute H(Y) - Entropy of output distribution
    non_zero_y = p_y > 0
    h_y_terms = p_y[non_zero_y] * np.log2(p_y[non_zero_y])
    h_y = -np.sum(h_y_terms)
    
    # Calculate mutual information using I(X;Y) = H(X) + H(Y) - H(XY)
    mutual = h_x + h_y - h_xy
    
    return mutual
