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



def ica(X, cycles, tol):
    '''Perform independent component analysis 
    Args:
        X (np.array): mixture matrix. Shape (nmix, time)
        cycles (int): number of max possible iterations 
        tol (float): convergence tolerance
    Returns:
        S_hat (np.array): predicted independent sources. Shape (nmix, time)
    '''
    # Step 1: Whiten the mixture matrix using the provided whiten function
    Z = whiten(X)
    nmix, _ = X.shape
    
    # Initialize unmixing matrix W
    W = np.zeros((nmix, nmix))
    
    for k in range(nmix):
        # Initialize random unit vector as starting point
        w_curr = np.random.randn(nmix)
        w_curr /= la.norm(w_curr)
        
        for _ in range(cycles):
            # Compute the projection of current w onto whitened data
            y = w_curr @ Z
            # Apply contrast function and its derivative
            g_y = np.tanh(y)
            dg_y = 1 - g_y ** 2
            
            # Calculate update terms using sample expectations
            term1 = np.mean(Z * g_y, axis=1)
            term2 = np.mean(dg_y) * w_curr
            w_new = term1 - term2
            
            # Orthogonalize against previously extracted components
            for i in range(k):
                w_new -= np.dot(w_new, W[i]) * W[i]
            
            # Normalize to unit norm, handle near-zero vectors to avoid division by zero
            norm = la.norm(w_new)
            if norm < 1e-10:
                w_new = np.random.randn(nmix)
                norm = la.norm(w_new)
                w_new /= norm
            else:
                w_new /= norm
            
            # Check convergence condition
            if np.abs(np.abs(np.dot(w_curr, w_new)) - 1) < tol:
                break
            
            # Update current vector for next iteration
            w_curr = w_new
        
        # Store the converged component as k-th row of W
        W[k] = w_curr
    
    # Compute predicted independent sources
    S_hat = W @ Z
    return S_hat
