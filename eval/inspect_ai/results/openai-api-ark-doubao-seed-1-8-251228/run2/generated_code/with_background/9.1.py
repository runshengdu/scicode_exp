import numpy as np



def WJ(A, b, eps, x_true, x0, omega):
    '''Solve a given linear system Ax=b with weighted Jacobi iteration method
    Input
    A:      N by N matrix, 2D array
    b:      N by 1 right hand side vector, 1D array
    eps:    Float number indicating error tolerance
    x_true: N by 1 true solution vector, 1D array
    x0:     N by 1 zero vector, 1D array
    omega:  float number shows weight parameter
    Output
    residuals: Float number shows L2 norm of residual (||Ax - b||_2)
    errors:    Float number shows L2 norm of error vector (||x-x_true||_2)
    '''
    # Convert inputs to numpy arrays for numerical operations
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    x_true = np.array(x_true, dtype=np.float64)
    x_prev = np.array(x0, dtype=np.float64).copy()
    
    while True:
        # Compute matrix-vector product Ax_prev
        Ax_prev = A @ x_prev
        # Compute residual vector b - Ax_prev
        res_prev = b - Ax_prev
        # Extract diagonal elements of A
        diag_A = np.diag(A)
        # Compute element-wise division of residual by diagonal elements
        temp = res_prev / diag_A
        # Update using weighted Jacobi formula (correct fixed-point iteration)
        x_current = x_prev + omega * temp
        # Calculate increment and its L2 norm
        delta = x_current - x_prev
        norm_delta = np.linalg.norm(delta, 2)
        
        # Check stopping condition
        if norm_delta < eps:
            break
        
        # Prepare for next iteration
        x_prev = x_current.copy()
    
    # Calculate final residual and error norms
    Ax_current = A @ x_current
    residual = np.linalg.norm(Ax_current - b, 2)
    error = np.linalg.norm(x_current - x_true, 2)
    
    return residual, error
