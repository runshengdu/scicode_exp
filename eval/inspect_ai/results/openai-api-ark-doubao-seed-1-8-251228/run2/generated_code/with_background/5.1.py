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
    Q = np.zeros((M, m+1), dtype=A.dtype)
    norm_b = np.linalg.norm(b)
    q1 = b / norm_b
    Q[:, 0] = q1
    
    if m == 0:
        return Q
    
    beta_prev = 0.0
    for k in range(m):
        q_k = Q[:, k]
        v = A @ q_k
        alpha = np.dot(q_k, v)
        v -= alpha * q_k
        if k > 0:
            v -= beta_prev * Q[:, k-1]
        beta = np.linalg.norm(v)
        Q[:, k+1] = v / beta
        beta_prev = beta
    
    return Q
