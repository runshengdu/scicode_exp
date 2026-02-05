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
