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


def mutual_info(channel, prior):
    '''Input
    channel: a classical channel, 2d array of floats; channel[i][j] means probability of i given j
    prior:   input random variable, 1d array of floats.
    Output
    mutual: mutual information between the input random variable and the random variable associated with the output of the channel, a single scalar value (float)
    '''
    # Convert inputs to numpy arrays
    channel_np = np.asarray(channel)
    prior_np = np.asarray(prior)
    
    # Compute H(X): entropy of the input prior distribution
    non_zero_prior = prior_np > 0
    h_x = -np.sum(prior_np[non_zero_prior] * np.log2(prior_np[non_zero_prior]))
    
    # Compute output distribution p_Y
    p_y = channel_np @ prior_np
    
    # Compute H(Y): entropy of the output distribution
    non_zero_py = p_y > 0
    h_y = -np.sum(p_y[non_zero_py] * np.log2(p_y[non_zero_py]))
    
    # Compute joint distribution p_XY
    p_xy = prior_np[:, np.newaxis] * channel_np.T
    
    # Compute H(XY): joint entropy of input and output
    non_zero_xy = p_xy > 0
    h_xy = -np.sum(p_xy[non_zero_xy] * np.log2(p_xy[non_zero_xy]))
    
    # Calculate mutual information using the entropy formula
    mutual = h_x + h_y - h_xy
    
    return mutual



def blahut_arimoto(channel, e):
    '''Input
    channel: a classical channel, 2d array of floats; Channel[i][j] means probability of i given j
    e:       error threshold, a single scalar value (float)
    Output
    rate_new: channel capacity, a single scalar value (float)
    '''


    channel_np = np.asarray(channel)
    n_inputs = channel_np.shape[1]
    
    # Initialize random prior distribution, normalized to sum to 1
    prior_old = np.random.rand(n_inputs)
    prior_old /= np.sum(prior_old)
    
    # Compute initial rate using KL divergence identity
    p_y = channel_np @ prior_old
    D_j = np.array([KL_divergence(channel_np[:, j], p_y) for j in range(n_inputs)])
    rate_old = np.sum(prior_old * D_j)
    
    while True:
        # Update prior distribution using the Blahut-Arimoto rule
        exp_terms = np.exp(D_j)
        numerator = prior_old * exp_terms
        Z = np.sum(numerator)
        prior_new = numerator / Z
        
        # Compute new rate
        p_y_new = channel_np @ prior_new
        D_j_new = np.array([KL_divergence(channel_np[:, j], p_y_new) for j in range(n_inputs)])
        rate_new = np.sum(prior_new * D_j_new)
        
        # Check convergence condition
        if np.abs(rate_new - rate_old) < e:
            break
        
        # Update variables for next iteration
        prior_old = prior_new
        rate_old = rate_new
        D_j = D_j_new
    
    return rate_new

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
