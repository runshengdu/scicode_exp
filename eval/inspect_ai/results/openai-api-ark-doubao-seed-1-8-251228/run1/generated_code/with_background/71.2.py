import numpy as np 
from scipy.optimize import fminbound
import itertools
from scipy.linalg import logm



def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''
    if not args:
        return np.array([[1.0]])
    
    processed = []
    for arr in args:
        if arr.ndim == 1:
            # Reshape 1D array to column vector (2D) and convert to float
            processed_arr = arr.reshape(-1, 1).astype(float)
        elif arr.ndim == 2:
            # Convert to float
            processed_arr = arr.astype(float)
        else:
            raise ValueError("All input arrays must be 1D or 2D.")
        processed.append(processed_arr)
    
    # Compute iterative Kronecker product
    result = processed[0]
    for current in processed[1:]:
        result = np.kron(result, current)
    
    return result
