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



def davidson_solver(matrixA, num_eigenvalues, threshold):
    '''Implements the Davidson algorithm to compute the first few eigenvalues of a symmetric matrix.
    Inputs:
    - matrixA: Symmetric matrix (2D array of float).
    - num_eigenvalues: Number of lowest eigenvalues to compute (int).
    - threshold: Convergence threshold for the algorithm (float).
    Output:
    - current_eigenvalues: computed eigenvalues (1D array of float).
    '''
    dim = matrixA.shape[0]
    # Validate input parameters
    assert matrixA.shape[0] == matrixA.shape[1], "Input matrix must be square."
    assert 0 < num_eigenvalues <= dim, "Number of eigenvalues must be between 1 and matrix dimension."
    
    # Initialize subspace with standard basis vectors for the first num_eigenvalues dimensions
    b = np.eye(dim, num_eigenvalues)
    
    while True:
        # Step 1: Project the matrix onto the subspace
        A_p = b.T @ matrixA @ b
        
        # Step 2: Diagonalize the projected symmetric matrix
        eig_vals_p, eig_vecs_p = np.linalg.eigh(A_p)
        
        # Select the smallest num_eigenvalues eigenvalues and corresponding eigenvectors
        selected_vals = eig_vals_p[:num_eigenvalues]
        selected_vecs = eig_vecs_p[:, :num_eigenvalues]
        
        converged = True
        correction_vecs = []
        
        for i in range(num_eigenvalues):
            lam = selected_vals[i]
            v = selected_vecs[:, i]
            
            # Compute approximate eigenvector
            x = b @ v
            
            # Calculate residual vector
            residual = matrixA @ x - lam * x
            res_norm = np.linalg.norm(residual)
            
            # Check convergence for this eigenvalue
            if res_norm > threshold:
                converged = False
            
            # Compute correction vector
            diag_A = np.diag(matrixA)
            correction = -residual / (diag_A - lam)
            correction_vecs.append(correction)
        
        # If all eigenvalues are converged, return the results
        if converged:
            return selected_vals
        
        # Append correction vectors to expand the subspace
        correction_mat = np.column_stack(correction_vecs)
        b = np.column_stack([b, correction_mat])
