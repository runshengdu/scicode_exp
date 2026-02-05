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
    
    # Process each input to ensure it's 2D
    processed_mats = []
    for mat in args:
        if mat.ndim == 1:
            # Reshape 1D vector to 2D column vector
            processed = mat.reshape(-1, 1)
        elif mat.ndim == 2:
            processed = mat
        else:
            raise ValueError("All inputs must be 1D (vectors) or 2D (matrices).")
        processed_mats.append(processed)
    
    # Compute iterative Kronecker product
    result = processed_mats[0]
    for mat in processed_mats[1:]:
        result = np.kron(result, mat)
    
    return result
