import numpy as np 
from scipy.optimize import fminbound
import itertools
from scipy.linalg import logm

def ket(dim, j):
    '''Input:
    dim: int or list, dimension(s) of the ket(s)
    j: int or list, the index(es) of the basis vector(s)
    Output:
    out: array of float, the matrix representation of the ket or tensor product of kets
    '''
    def _single_ket(d, idx):
        vec = np.zeros(d, dtype=float)
        vec[idx] = 1.0
        return vec
    
    # Case 1: Single basis vector (dim is int, j is int)
    if isinstance(dim, int) and isinstance(j, int):
        return _single_ket(dim, j)
    
    # Case 2: Tensor product of multiple kets with same dimension (dim is int, j is list)
    elif isinstance(dim, int) and isinstance(j, list):
        if not all(isinstance(idx, int) for idx in j):
            raise TypeError("All elements in j must be integers when dim is an integer")
        kets = [_single_ket(dim, idx) for idx in j]
        result = kets[0]
        for vec in kets[1:]:
            result = np.kron(result, vec)
        return result
    
    # Case 3: Tensor product of kets with different dimensions (dim is list, j is list)
    elif isinstance(dim, list) and isinstance(j, list):
        if len(dim) != len(j):
            raise ValueError("dim and j must have the same length when both are lists")
        if not all(isinstance(d, int) for d in dim):
            raise TypeError("All elements in dim must be integers")
        if not all(isinstance(idx, int) for idx in j):
            raise TypeError("All elements in j must be integers")
        kets = [_single_ket(d, idx) for d, idx in zip(dim, j)]
        result = kets[0]
        for vec in kets[1:]:
            result = np.kron(result, vec)
        return result
    
    # Invalid input combinations
    else:
        raise TypeError(
            "Invalid input combination: "
            "dim must be int or list; "
            "if dim is int, j can be int or list; "
            "if dim is list, j must be list of same length"
        )



def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''
    processed = []
    for arg in args:
        arr = np.asarray(arg)
        if arr.ndim not in (1, 2):
            raise ValueError("All inputs must be 1D (vectors) or 2D (matrices)")
        arr = arr.astype(float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        processed.append(arr)
    
    if not processed:
        return np.array([[1.0]], dtype=float)
    
    result = processed[0]
    for mat in processed[1:]:
        result = np.kron(result, mat)
    
    return result
