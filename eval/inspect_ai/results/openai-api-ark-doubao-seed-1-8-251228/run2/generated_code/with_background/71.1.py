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
