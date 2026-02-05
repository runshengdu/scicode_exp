import numpy as np


def basis(i, p, M, h, etype):
    '''Inputs
    i: int, the index of node (1-based)
    p: array of arbitrary size 1,2, or 3, the coordinates to evaluate
    M: int, the total number of the nodal dofs
    h: int, the element size
    etype: int, basis function type; When type equals to 1, 
    it returns $\omega^1(x)$, when the type equals to 2, it returns the value of function $\omega^2(x)$.
    Outputs
    v: array of size 1,2, or 3, value of basis function
    '''
    v = np.zeros_like(p, dtype=np.float64)
    x_nodes = np.arange(M) * h  # Nodes are at 0, h, 2h, ..., (M-1)h
    
    if etype == 1:
        # Compute ω_i^1(x)
        if i == 1:
            # Boundary node, no left element, always 0
            pass
        else:
            x_prev = x_nodes[i-2]  # x_{i-1}
            x_curr = x_nodes[i-1]  # x_i
            mask = (p >= x_prev) & (p <= x_curr)
            v[mask] = (p[mask] - x_prev) / h
    elif etype == 2:
        # Compute ω_i^2(x)
        if i == M:
            # Boundary node, no right element, always 0
            pass
        else:
            x_curr = x_nodes[i-1]  # x_i
            x_next = x_nodes[i]    # x_{i+1}
            mask = (p >= x_curr) & (p <= x_next)
            v[mask] = (x_next - p[mask]) / h
    
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
        b = np.zeros((1, 1), dtype=np.float64)
        return A, b
    
    # Element size for domain [0, 1]
    h = 1.0 / (M - 1)
    # Third-order Gauss quadrature points and weights
    gauss_xi = np.array([-3/5, 0, 3/5], dtype=np.float64)
    gauss_w = np.array([5/9, 8/9, 5/9], dtype=np.float64)
    # Coefficients from the weak form (advection and diffusion)
    a = 1.0
    kappa = 1.0
    
    # Initialize global matrix and right-hand side vector
    A = np.zeros((M, M), dtype=np.float64)
    b = np.zeros((M, 1), dtype=np.float64)
    
    # Iterate over each element (0-based index)
    for e in range(M - 1):
        x_e = e * h          # Left boundary of current element
        x_e1 = (e + 1) * h   # Right boundary of current element
        
        # Compute local stiffness matrix using Gauss quadrature
        local_K = np.zeros((2, 2), dtype=np.float64)
        for xi, w in zip(gauss_xi, gauss_w):
            # Map reference coordinate to physical coordinate
            x_g = (x_e1 - x_e) / 2 * xi + (x_e + x_e1) / 2
            # Derivatives of local basis functions (constant over element)
            dN0_dx = -1.0 / h
            dN1_dx = 1.0 / h
            # Local basis function values at Gauss point
            N0 = (x_e1 - x_g) / h
            N1 = (x_g - x_e) / h
            # Derivatives of local basis functions (constant values)
            dN0_dx_val = -1.0 / h
            dN1_dx_val = 1.0 / h
            
            # Calculate integrand terms for each local matrix entry
            term_K00 = dN0_dx * (-a * N0 + kappa * dN0_dx_val)
            term_K01 = dN0_dx * (-a * N1 + kappa * dN1_dx_val)
            term_K10 = dN1_dx * (-a * N0 + kappa * dN0_dx_val)
            term_K11 = dN1_dx * (-a * N1 + kappa * dN1_dx_val)
            
            # Weighted contribution from current Gauss point
            weight = (x_e1 - x_e) / 2 * w
            local_K[0, 0] += weight * term_K00
            local_K[0, 1] += weight * term_K01
            local_K[1, 0] += weight * term_K10
            local_K[1, 1] += weight * term_K11
        
        # Accumulate local matrix into global matrix
        A[e:e+2, e:e+2] += local_K
        
        # Compute local contributions to right-hand side using Gauss quadrature
        b0 = 0.0  # Contribution to left node of element
        b1 = 0.0  # Contribution to right node of element
        for xi, w in zip(gauss_xi, gauss_w):
            x_g = (x_e1 - x_e) / 2 * xi + (x_e + x_e1) / 2
            # Local basis function values at Gauss point
            N0 = (x_e1 - x_g) / h
            N1 = (x_g - x_e) / h
            # Integrand for right-hand side
            integrand_b0 = 12 * (x_g ** 2) * N0
            integrand_b1 = 12 * (x_g ** 2) * N1
            # Weighted contribution from current Gauss point
            weight = (x_e1 - x_e) / 2 * w
            b0 += weight * integrand_b0
            b1 += weight * integrand_b1
        
        # Accumulate local contributions into global right-hand side
        b[e, 0] += b0
        b[e+1, 0] += b1
    
    return A, b
