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
    # Calculate mean of each row, keeping dimensions for broadcasting
    row_means = np.mean(X, axis=1, keepdims=True)
    D = X - row_means
    
    if divide_sd:
        # Calculate standard deviation of each centered row
        row_stds = np.std(D, axis=1, keepdims=True)
        # Scale the centered matrix by row-wise standard deviations
        D = D / row_stds
    
    return D



def whiten(X):
    '''Whiten matrix X
    Args: 
        X (np.array): mixture matrix. Shape (nmix, time)
    Return:
        Z (np.array): whitened matrix. Shape (nmix, time)
    '''
    # Step 1: Center the input matrix along rows (subtract row-wise means)
    D = center(X, divide_sd=False)
    
    # Step 2: Compute sample covariance matrix of the centered data
    cov_mat = np.cov(D)
    
    # Step 3: Eigenvalue decomposition of the covariance matrix
    eig_vals, eig_vecs = la.eigh(cov_mat)
    
    # Step 4: Construct whitening matrix with numerical stability
    eps = 1e-10
    whitening_matrix = eig_vecs @ np.diag(1.0 / np.sqrt(eig_vals + eps)) @ eig_vecs.T
    
    # Step 5: Apply whitening transformation to centered data
    Z = whitening_matrix @ D
    
    return Z
