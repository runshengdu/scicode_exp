import numpy as np
from scipy.linalg import sqrtm
import itertools



def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''
    if not args:
        raise ValueError("At least one matrix or vector must be provided.")
    
    processed_arrays = []
    for arg in args:
        arr = np.asarray(arg)
        if arr.ndim == 1:
            # Convert 1D vector to 2D column vector
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            raise ValueError("All inputs must be 1D (vectors) or 2D (matrices).")
        processed_arrays.append(arr)
    
    # Compute successive Kronecker products
    result = processed_arrays[0]
    for mat in processed_arrays[1:]:
        result = np.kron(result, mat)
    
    return result
