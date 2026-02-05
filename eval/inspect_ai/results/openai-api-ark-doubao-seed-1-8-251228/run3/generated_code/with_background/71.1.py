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
