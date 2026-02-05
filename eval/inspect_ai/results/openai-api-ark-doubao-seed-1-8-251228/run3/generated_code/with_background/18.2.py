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



def NURBS_2D(xi_1, xi_2, i_1, i_2, p_1, p_2, n_1, n_2, Xi_1, Xi_2, w):
    '''Inputs:
    xi_1 : parameter coordinate at the first dof, float
    xi_2 : parameter coordinate at the second dof, float
    i_1 : index of the basis function to be evaluated at the first dof, integer
    i_2 : index of the basis function to be evaluated at the second dof, integer
    p_1 : polynomial degree of the basis function at the first dof, integer
    p_2 : polynomial degree of the basis function at the second dof, integer
    n_1 : total number of basis function at the first dof, integer
    n_2 : total number of basis function at the second dof, integer
    Xi_1 : knot vector of the first dof, 1d array of arbitrary size
    Xi_2 : knot vector of the second dof, 1d array of arbitrary size
    w : array storing NURBS weights, 1d array
    Outputs:
    N : value of the basis functions evaluated at the given paramter coordinates, 1d array of size 1 or 2
    '''
    # Compute B-spline basis functions for the specified indices
    N_i1 = Bspline(xi_1, i_1, p_1, Xi_1)
    N_i2 = Bspline(xi_2, i_2, p_2, Xi_2)
    
    # Check if indices are within valid range
    if not (0 <= i_1 < n_1) or not (0 <= i_2 < n_2):
        # Return zero array matching the broadcasted shape of inputs
        broadcasted = N_i1 * N_i2
        return np.zeros_like(broadcasted, dtype=np.float64).ravel()
    
    # Get the corresponding weight for the control point (i_1, i_2)
    weight_idx = i_1 * n_2 + i_2
    if weight_idx < 0 or weight_idx >= len(w):
        return np.zeros_like(N_i1 * N_i2, dtype=np.float64).ravel()
    w_ij = w[weight_idx]
    
    # Calculate numerator of the rational basis function
    numerator = N_i1 * N_i2 * w_ij
    
    # Calculate denominator (weighted sum of all basis function products)
    denominator = np.zeros_like(numerator, dtype=np.float64)
    for j1 in range(n_1):
        n1 = Bspline(xi_1, j1, p_1, Xi_1)
        for j2 in range(n_2):
            n2 = Bspline(xi_2, j2, p_2, Xi_2)
            w_j = w[j1 * n_2 + j2]
            denominator += n1 * n2 * w_j
    
    # Handle division by zero and compute final result
    mask = (denominator == 0.0)
    result = np.where(mask, 0.0, numerator / denominator)
    
    # Ensure output is a 1D array
    return result.ravel()
