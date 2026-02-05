import numpy as np



def lanczos(A, b, m):
    '''Inputs:
    A : Matrix, 2d array of arbitrary size M * M
    b : Vector, 1d array of arbitrary size M * 1
    m : integer, m < M
    Outputs:
    Q : Matrix, 2d array of size M*(m+1)
    '''
    M = A.shape[0]
    # Initialize Q matrix with M rows and m+1 columns
    Q = np.zeros((M, m + 1))
    
    # Normalize the initial vector to get the first column of Q
    b_1d = b.ravel()
    norm_b = np.linalg.norm(b_1d)
    q0 = b_1d / norm_b
    Q[:, 0] = q0
    
    # Return immediately if no iterations are needed
    if m == 0:
        return Q
    
    # Initialize iteration variables
    q_prev = q0  # Current orthogonal vector q_k
    q_prev_prev = np.zeros(M)  # Previous orthogonal vector q_{k-1}
    beta_prev = 0.0  # Previous beta value beta_{k-1}
    
    for k in range(m):
        # Compute alpha_k = q_k^T A q_k
        alpha = q_prev.T @ A @ q_prev
        
        # Compute residual vector
        r = A @ q_prev - alpha * q_prev - beta_prev * q_prev_prev
        
        # Compute beta_k as the norm of the residual
        beta = np.linalg.norm(r)
        
        # Normalize residual to get next orthogonal vector q_{k+1}
        q_next = r / beta
        
        # Store the new orthogonal vector in Q
        Q[:, k + 1] = q_next
        
        # Update variables for next iteration
        q_prev_prev, q_prev = q_prev, q_next
        beta_prev = beta
    
    return Q
