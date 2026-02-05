import numpy as np


def basis(i, p, M, h, etype):
    '''Inputs
    i: int, the index of node (1-based)
    p: array of arbitrary size 1,2, or 3, the coordinates to evaluate
    M: int, the total number of the nodal dofs
    h: float, the element size (uniform)
    etype: int, basis function type; When type equals to 1, 
    it returns $\omega^1(x)$, when the type equals to 2, it returns the value of function $\omega^2(x)$.
    Outputs
    v: array of size 1,2, or 3, value of basis function
    '''
    p = np.asarray(p, dtype=np.float64)
    v = np.zeros_like(p)
    
    if etype == 1:
        # Compute ω_i^1(x): active on [x_{i-1}, x_i] for interior nodes i>1, else 0
        if i > 1 and i <= M:
            x_left = (i - 2) * h  # x_{i-1} (1-based node i-1 corresponds to (i-2)*h)
            x_right = (i - 1) * h  # x_i (1-based node i corresponds to (i-1)*h)
            mask = (p >= x_left) & (p <= x_right)
            v[mask] = (p[mask] - x_left) / h
        # For i=1, ω_1^1(x) is 0 everywhere, so v remains 0
    elif etype == 2:
        # Compute ω_i^2(x): active on [x_i, x_{i+1}] for interior nodes i<M, else 0
        if i >= 1 and i < M:
            x_left = (i - 1) * h  # x_i (1-based node i corresponds to (i-1)*h)
            x_right = i * h        # x_{i+1} (1-based node i+1 corresponds to i*h)
            mask = (p >= x_left) & (p <= x_right)
            v[mask] = (x_right - p[mask]) / h
        # For i=M, ω_M^2(x) is 0 everywhere, so v remains 0
    else:
        raise ValueError("etype must be either 1 or 2")
    
    return v


def assemble(M):
    '''Inputs:
    M : number of grid, integer
    Outputs:
    A: mass matrix, 2d array size M*M
    b: right hand side vector , 1d array size M*1
    '''
    if M == 1:
        A = np.zeros((1, 1), dtype=np.float64)
        b = np.zeros((1,), dtype=np.float64)
        return A, b
    
    h = 1.0 / (M - 1)
    A = np.zeros((M, M), dtype=np.float64)
    b = np.zeros(M, dtype=np.float64)
    
    # Gaussian quadrature points and weights as specified
    xi = np.array([-3/5, 0, 3/5], dtype=np.float64)
    weights = np.array([5/9, 8/9, 5/9], dtype=np.float64)
    
    # Advection and diffusion coefficients (assumed 1.0 as problem statement didn't specify)
    a_coeff = 1.0
    kappa_coeff = 1.0
    
    # Iterate over each element (1-based index from 1 to M-1)
    for e in range(1, M):
        a_elem = (e - 1) * h  # Left boundary of current element
        b_elem = e * h         # Right boundary of current element
        
        # Map reference Gaussian points to physical domain
        x_g = (b_elem - a_elem) / 2 * xi + (a_elem + b_elem) / 2
        
        # Compute contributions to right-hand side vector b
        # Contribution from node e (1-based)
        omega_e = (b_elem - x_g) / h
        integrand_b_e = 12 * x_g ** 2 * omega_e
        integral_b_e = (h / 2) * np.sum(weights * integrand_b_e)
        b[e-1] += integral_b_e
        
        # Contribution from node e+1 (1-based)
        omega_e1 = (x_g - a_elem) / h
        integrand_b_e1 = 12 * x_g ** 2 * omega_e1
        integral_b_e1 = (h / 2) * np.sum(weights * integrand_b_e1)
        b[e] += integral_b_e1
        
        # Derivatives of basis functions on current element
        omega_e_x = -1.0 / h
        omega_e1_x = 1.0 / h
        
        # Compute contributions to mass matrix A
        # (i=e, j=e)
        term = omega_e_x * (-a_coeff * omega_e + kappa_coeff * omega_e_x)
        integral = (h / 2) * np.sum(weights * term)
        A[e-1, e-1] += integral
        
        # (i=e, j=e+1)
        term = omega_e_x * (-a_coeff * omega_e1 + kappa_coeff * omega_e1_x)
        integral = (h / 2) * np.sum(weights * term)
        A[e-1, e] += integral
        
        # (i=e+1, j=e)
        term = omega_e1_x * (-a_coeff * omega_e + kappa_coeff * omega_e_x)
        integral = (h / 2) * np.sum(weights * term)
        A[e, e-1] += integral
        
        # (i=e+1, j=e+1)
        term = omega_e1_x * (-a_coeff * omega_e1 + kappa_coeff * omega_e1_x)
        integral = (h / 2) * np.sum(weights * term)
        A[e, e] += integral
    
    return A, b



def stabilization(A, b):
    '''Inputs:
    A : mass matrix, 2d array of shape (M,M)
    b : right hand side vector, 1d array of shape (M,)
    Outputs:
    A : mass matrix, 2d array of shape (M,M)
    b : right hand side vector 1d array of any size, 1d array of shape (M,)
    '''

    
    M = A.shape[0]
    if M == 1:
        return A.copy(), b.copy()
    
    # Create copies to avoid modifying original input arrays
    A_new = A.copy()
    b_new = b.copy()
    
    # Define parameters
    s_kappa = 1.0
    a_coeff = 200.0
    C = 50.0
    kappa_coeff = 1.0
    h = 1.0 / (M - 1) if M > 1 else 0.0
    
    # Calculate V_kappa
    V_kappa = C * (1.0 / h) * (1 + abs(s_kappa))
    
    # Calculate element Peclet number
    P_e = (abs(a_coeff) * h) / (2 * kappa_coeff)
    
    # Calculate tau parameter
    if P_e == 0:
        tau = 0.0
    else:
        # Use cosh/sinh for compatibility with older numpy versions
        coth_Pe = np.cosh(P_e) / np.sinh(P_e)
        tau = (h / (2 * abs(a_coeff))) * (coth_Pe - 1.0 / P_e)
    
    # Construct G matrix for SUPG (integral of basis function derivatives product)
    G = np.zeros_like(A_new)
    # Diagonal elements
    diag_vals = np.full(M, 2.0 / h, dtype=np.float64)
    diag_vals[0] = 1.0 / h
    diag_vals[-1] = 1.0 / h
    np.fill_diagonal(G, diag_vals)
    # Off-diagonal elements
    if M >= 2:
        off_diag_vals = np.full(M-1, -1.0 / h, dtype=np.float64)
        G[np.arange(M-1), np.arange(1, M)] = off_diag_vals
        G[np.arange(1, M), np.arange(M-1)] = off_diag_vals
    
    # Compute SUPG matrix contribution
    SUPG_matrix = (a_coeff ** 2) * tau * G
    A_new += SUPG_matrix
    
    # Compute Nitsche method matrix contributions
    Nitsche_matrix = np.zeros_like(A_new)
    if M >= 2:
        # Contributions from last node (i = M-1, 0-based)
        i_last = M - 1
        # Coefficient for second last node (j = M-2)
        Nitsche_matrix[i_last, M-2] += kappa_coeff / h
        # Coefficient for last node (j = M-1)
        coeff_last = a_coeff - (kappa_coeff / h) - (s_kappa * kappa_coeff) / h + V_kappa
        Nitsche_matrix[i_last, i_last] += coeff_last
        
        # Contributions from second last node (i = M-2, 0-based)
        i_second_last = M - 2
        Nitsche_matrix[i_second_last, i_last] += (s_kappa * kappa_coeff) / h
    
    A_new += Nitsche_matrix
    
    # Compute SUPG vector contribution
    SUPG_vector = np.zeros_like(b_new)
    # First node contribution
    SUPG_vector[0] = 4 * a_coeff * tau * (h ** 2)
    # Last node contribution
    last_term = 12 - 12 * h + 4 * (h ** 2)
    SUPG_vector[-1] = -a_coeff * tau * last_term
    # Interior nodes contribution
    mid_indices = np.arange(1, M-1)
    SUPG_vector[mid_indices] = 24 * a_coeff * tau * mid_indices * (h ** 2)
    
    # Compute Nitsche vector contribution
    Nitsche_vector = np.zeros_like(b_new)
    if M >= 2:
        i_last = M - 1
        Nitsche_vector[i_last] = (s_kappa * kappa_coeff) / h - V_kappa
        i_second_last = M - 2
        Nitsche_vector[i_second_last] = - (s_kappa * kappa_coeff) / h
    
    # Update right-hand side vector
    b_new = b_new - SUPG_vector - Nitsche_vector
    
    return A_new, b_new
