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
    assert matrixA.shape[1] == dim, "Matrix must be square."
    assert 1 <= num_eigenvalues <= dim, "Number of eigenvalues must be between 1 and matrix dimension."
    
    # Initialize subspace basis with random orthonormal vectors
    b = np.random.normal(size=(dim, num_eigenvalues))
    b, _ = np.linalg.qr(b)  # Orthonormalize using QR decomposition
    
    max_iter = 1000
    iter_count = 0
    
    while iter_count < max_iter:
        # Project matrix onto subspace: H = b^T A b
        H = b.T @ matrixA @ b
        
        # Diagonalize projected symmetric matrix
        eigenvalues, eigenvectors = np.linalg.eigh(H)  # Eigenvalues sorted ascending
        
        # Extract relevant eigenvalues and eigenvectors
        current_eigenvalues = eigenvalues[:num_eigenvalues]
        current_eigenvectors = eigenvectors[:, :num_eigenvalues]
        
        # Compute residues and check convergence
        residues = []
        converged = True
        for i in range(num_eigenvalues):
            # Compute Ritz vector
            u = b @ current_eigenvectors[:, i]
            # Compute residue vector
            r = matrixA @ u - current_eigenvalues[i] * u
            residues.append(r)
            # Check convergence criterion
            if np.linalg.norm(r) > threshold:
                converged = False
        
        if converged:
            return current_eigenvalues
        
        # Generate correction vectors
        correction_vectors = []
        diag_A = np.diag(matrixA)
        for i in range(num_eigenvalues):
            r = residues[i]
            lam = current_eigenvalues[i]
            # Compute denominator with division-by-zero protection
            denominator = diag_A - lam
            mask = np.abs(denominator) < 1e-12
            denominator[mask] = 1e-12
            # Compute correction vector
            q = -r / denominator
            # Orthonormalize against existing subspace (b is orthonormal)
            proj = b @ (b.T @ q)
            q -= proj
            # Normalize and filter negligible vectors
            norm_q = np.linalg.norm(q)
            if norm_q > 1e-12:
                q /= norm_q
                correction_vectors.append(q)
        
        # Update subspace basis if we have valid corrections
        if correction_vectors:
            # Orthonormalize corrections against each other
            corrections = np.column_stack(correction_vectors)
            corrections, _ = np.linalg.qr(corrections)
            # Append to basis
            b = np.hstack([b, corrections])
        
        iter_count += 1
    
    # Return best approximation if max iterations reached
    return current_eigenvalues
