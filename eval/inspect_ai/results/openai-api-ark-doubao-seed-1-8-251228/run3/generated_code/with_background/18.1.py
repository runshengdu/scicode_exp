import numpy as np



def Bspline(xi, i, p, Xi):
    '''Inputs:
    xi : knot index, integer
    i : polynomial index , integer
    p : polynomial degree of basis function , integer
    Xi : knot vector, 1d array of arbitrary size
    Outputs:
    1d array of size 1，2 or 3
    '''
    xi_arr = np.atleast_1d(xi)
    if p == 0:
        # Base case: degree 0 basis function
        if i < 0 or i + 1 >= len(Xi):
            return np.zeros_like(xi_arr, dtype=np.float64)
        mask = (xi_arr >= Xi[i]) & (xi_arr < Xi[i+1])
        result = np.where(mask, 1.0, 0.0)
        return result
    else:
        # Recursive case: compute alpha for first term
        if i < 0 or i + p >= len(Xi):
            alpha = 0.0
        else:
            denom_alpha = Xi[i + p] - Xi[i]
            alpha = (xi_arr - Xi[i]) / denom_alpha if denom_alpha != 0 else 0.0
        
        # Compute beta for second term
        if (i + 1) < 0 or (i + 1) + p >= len(Xi):
            beta = 0.0
        else:
            denom_beta = Xi[i + p + 1] - Xi[i + 1]
            beta = (Xi[i + p + 1] - xi_arr) / denom_beta if denom_beta != 0 else 0.0
        
        # Recursive call and combine terms
        term1 = alpha * Bspline(xi, i, p-1, Xi)
        term2 = beta * Bspline(xi, i+1, p-1, Xi)
        return term1 + term2
