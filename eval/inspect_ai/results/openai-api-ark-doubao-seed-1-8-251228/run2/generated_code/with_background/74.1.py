import numpy as np



def householder(A):
    '''Inputs:
    A : Matrix of size m*n, m>=n
    Outputs:
    A : Matrix of size m*n
    '''
    m, n = A.shape
    for k in range(n):
        # Extract the subvector from column k starting at row k
        x = A[k:, k]
        m_k = len(x)
        
        # Skip if there are no elements below to zero out
        if m_k == 1:
            continue
        
        # Skip if elements below the diagonal are already zero
        if np.linalg.norm(x[1:]) == 0:
            continue
        
        norm_x = np.linalg.norm(x)
        # Skip if the subvector is already zero
        if norm_x == 0:
            continue
        
        # Compute the sign of the first element for numerical stability
        s = np.copysign(1, x[0])
        
        # Create the first standard basis vector of the same size as x
        e1 = np.zeros_like(x)
        e1[0] = 1.0
        
        # Compute the Householder vector v
        v = -s * norm_x * e1 - x
        
        # Compute v^T v (scalar)
        vt_v = v.T @ v
        
        # Compute v^T multiplied by the submatrix A[k:, k:]
        vt_B = v.T @ A[k:, k:]
        
        # Apply the Householder reflector to the submatrix
        A[k:, k:] -= 2.0 * np.outer(v, vt_B) / vt_v
    
    return A
