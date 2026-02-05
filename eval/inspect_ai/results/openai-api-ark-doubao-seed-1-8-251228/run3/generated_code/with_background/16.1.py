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
    # Create diagonal matrix with increasing values 1, 2, ..., dim on the diagonal
    diag_matrix = np.diag(np.arange(1, dim + 1))
    # Generate matrix of normally distributed random values scaled by noise
    noisy_matrix = noise * np.random.normal(size=(dim, dim))
    # Combine the diagonal matrix and noisy matrix
    M = diag_matrix + noisy_matrix
    # Symmetrize by averaging the matrix with its transpose
    A = (M + M.T) / 2.0
    return A
