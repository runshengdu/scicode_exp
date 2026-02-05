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



def ica(X, cycles, tol):
    '''Perform independent component analysis 
    Args:
        X (np.array): mixture matrix. Shape (nmix, time)
        cycles (int): number of max possible iterations 
        tol (float): convergence tolerance
    Returns:
        S_hat (np.array): predicted independent sources. Shape (nmix, time)
    '''
    # Whiten the input matrix
    Z = whiten(X)
    m, T = Z.shape
    W = np.zeros((m, m))
    
    for k in range(m):
        # Initialize random unit vector
        w = np.random.randn(m, 1)
        w /= la.norm(w)
        
        for _ in range(cycles):
            # Compute g(w^T Z)
            g = np.tanh(w.T @ Z)
            # First term: E[Z * g(w^T Z)]
            term1 = Z @ g.T / T
            # Compute dg/dx and its mean
            dg = 1 - g ** 2
            mean_dg = np.mean(dg)
            # Update rule
            w_new = term1 - w * mean_dg
            
            # Orthogonalize against already found components
            if k > 0:
                projections = W[:k, :] @ w_new
                w_new -= W[:k, :].T @ projections
            
            # Normalize to unit norm
            norm = la.norm(w_new)
            if norm < 1e-10:
                # Reinitialize if norm is too small to avoid division by zero
                w = np.random.randn(m, 1)
                w /= la.norm(w)
                continue
            w_new /= norm
            
            # Check convergence
            dot_product = w.T @ w_new
            if np.abs(np.abs(dot_product) - 1) < tol:
                break
            
            # Update current vector
            w = w_new
        
        # Assign the converged vector as the k-th row of W
        W[k, :] = w.T.ravel()
    
    # Compute predicted sources
    S_hat = W @ Z
    return S_hat
