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
    # Extract diagonal elements of A
    diag_A = np.diag(A)
    # Initialize previous iterate with initial guess
    x_prev = x0.copy()
    
    while True:
        # Compute residual vector
        r = b - A @ x_prev
        # Calculate next iterate using weighted Jacobi formula
        x_next = omega * (r / diag_A + x_prev)
        # Compute L2 norm of the increment
        delta_norm = np.linalg.norm(x_next - x_prev)
        # Check stopping condition
        if delta_norm < eps:
            break
        # Update previous iterate for next iteration
        x_prev = x_next.copy()
    
    # Calculate final residual and error
    x_final = x_next
    residual = np.linalg.norm(A @ x_final - b)
    error = np.linalg.norm(x_final - x_true)
    
    return residual, error
