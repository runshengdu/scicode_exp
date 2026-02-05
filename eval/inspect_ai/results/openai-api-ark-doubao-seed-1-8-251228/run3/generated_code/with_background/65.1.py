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
    
    # Convert 1D inputs to 2D column vectors and validate input dimensions
    processed = []
    for arg in args:
        if arg.ndim == 1:
            processed.append(arg.reshape(-1, 1))
        elif arg.ndim == 2:
            processed.append(arg)
        else:
            raise ValueError("All inputs must be 1D vectors or 2D matrices.")
    
    # Compute iterative Kronecker product leveraging associativity
    result = processed[0]
    for mat in processed[1:]:
        result = np.kron(result, mat)
    
    return result
