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
