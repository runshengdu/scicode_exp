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
