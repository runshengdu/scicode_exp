import numpy as np
import numpy.linalg as la
from scipy import signal

def center(X, divide_sd=True):
    '''Center the input matrix X and optionally scale it by the standard deviation.
    Args:
        X (np.ndarray): The input matrix of shape (nmix, time).
        divide_sd (bool): If True, divide by the standard deviation. Defaults to True.
    Returns:
        np.ndarray: The centered (and optionally scaled) matrix of the same shape as the input.
    '''
    # Calculate row-wise means with keepdims to maintain broadcasting compatibility
    row_means = X.mean(axis=1, keepdims=True)
    # Center the matrix by subtracting row means
    D = X - row_means
    
    if divide_sd:
        # Calculate row-wise standard deviations with keepdims
        row_stds = X.std(axis=1, keepdims=True)
        # Scale the centered matrix by row standard deviations
        D = D / row_stds
    
    return D



def whiten(X):
    '''Whiten matrix X
    Args: 
        X (np.array): mixture matrix. Shape (nmix, time)
    Return:
        Z (np.array): whitened matrix. Shape (nmix, time)
    '''
    # Center the matrix along rows (subtract mean, no scaling)
    D = center(X, divide_sd=False)
    
    # Singular Value Decomposition of the centered data
    U, S, Vh = la.svd(D, full_matrices=False)
    
    # Number of time points
    n_time = D.shape[1]
    
    # Compute scaling factor to ensure unit covariance, add epsilon to avoid division by zero
    epsilon = 1e-10
    scaling = np.sqrt(n_time - 1) / (S + epsilon)
    
    # Apply whitening transformation
    Z = (U.T @ D) * scaling[:, np.newaxis]
    
    return Z
