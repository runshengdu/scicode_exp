import numpy as np 
from scipy.optimize import fminbound
import itertools
from scipy.linalg import logm


def ket(dim, j):
    '''Input:
    dim: int or list, dimension of the ket
    j: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    '''
    if isinstance(j, int):
        if isinstance(dim, int):
            out = np.zeros(dim, dtype=float)
            out[j] = 1.0
        else:
            raise TypeError("When j is an integer, dim must be an integer")
    elif isinstance(j, list):
        if isinstance(dim, int):
            vectors = []
            for idx in j:
                vec = np.zeros(dim, dtype=float)
                vec[idx] = 1.0
                vectors.append(vec)
            out = vectors[0]
            for vec in vectors[1:]:
                out = np.kron(out, vec)
        elif isinstance(dim, list):
            if len(j) != len(dim):
                raise ValueError("Length of j must match length of dim when both are lists")
            vectors = []
            for idx, d in zip(j, dim):
                vec = np.zeros(d, dtype=float)
                vec[idx] = 1.0
                vectors.append(vec)
            out = vectors[0]
            for vec in vectors[1:]:
                out = np.kron(out, vec)
        else:
            raise TypeError("dim must be an integer or list of integers")
    else:
        raise TypeError("j must be an integer or list of integers")
    
    out = out.reshape(-1, 1)
    return out


def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''
    if len(args) == 0:
        return np.array([[1.0]], dtype=float)
    
    processed = []
    for arg in args:
        if not isinstance(arg, np.ndarray):
            raise TypeError("All inputs must be numpy ndarrays")
        if arg.ndim > 2:
            raise ValueError("Inputs must be 0D (scalars), 1D (vectors) or 2D (matrices)")
        
        if arg.ndim == 0:
            arg_2d = arg.reshape(1, 1)
        elif arg.ndim == 1:
            arg_2d = arg.reshape(-1, 1)
        else:
            arg_2d = arg
        
        processed.append(arg_2d.astype(float))
    
    result = processed[0]
    for mat in processed[1:]:
        result = np.kron(result, mat)
    
    return result


