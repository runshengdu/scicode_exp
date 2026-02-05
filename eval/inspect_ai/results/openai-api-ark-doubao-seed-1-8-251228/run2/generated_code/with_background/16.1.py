import math
import numpy as np



def init_matrix(dim, noise):
    '''Generate a symmetric matrix with increasing values along its diagonal.
    Inputs:
    - dim: The dimension of the matrix (int).
    - noise: Noise level (float).
    Output:
    - A: a 2D array where each element is a float, representing the symmetric matrix.
    '''
    # Create base matrix with increasing diagonal elements (1, 2, ..., dim)
    base = np.diag(np.arange(1, dim + 1))
    # Generate noise matrix with elements as noise multiplied by standard normal samples
    noise_mat = noise * np.random.randn(dim, dim)
    # Combine base matrix with noise
    M = base + noise_mat
    # Symmetrize the matrix by averaging with its transpose
    A = (M + M.T) / 2
    return A
