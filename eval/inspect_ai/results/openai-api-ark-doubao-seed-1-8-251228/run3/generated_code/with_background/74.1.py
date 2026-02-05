import numpy as np



def householder(A):
    '''Inputs:
    A : Matrix of size m*n, m>=n
    Outputs:
    A : Matrix of size m*n
    '''
    # Create a copy to avoid modifying the original matrix
    A = np.asarray(A).copy()
    m, n = A.shape
    
    for k in range(n):
        # Extract the subvector from column k starting at row k
        x = A[k:, k]
        # Compute L2 norm of the subvector
        norm_x = np.linalg.norm(x)
        
        # Skip if the subvector is already effectively zero
        if norm_x < 1e-12:
            continue
        
        # Determine the sign of the first element for numerical stability
        if x[0].real >= 0:
            sign_x1 = 1.0
        else:
            sign_x1 = -1.0
        
        # Create the standard basis vector e1 (first element 1, rest 0)
        e1 = np.zeros_like(x)
        e1[0] = 1.0
        
        # Compute the Householder vector v
        v = -sign_x1 * norm_x * e1 - x
        
        # Compute v^H * v (inner product of v with its conjugate)
        v_star_v = np.dot(v.conj(), v)
        
        # Compute v^H multiplied by the relevant submatrix of A
        v_star_B = np.dot(v.conj().T, A[k:, k:])
        
        # Calculate the update matrix to apply the Householder reflection
        update = 2 * np.outer(v, v_star_B) / v_star_v
        
        # Apply the update to transform the submatrix
        A[k:, k:] -= update
    
    return A
