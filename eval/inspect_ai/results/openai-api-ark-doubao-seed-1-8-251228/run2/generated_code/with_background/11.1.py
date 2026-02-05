import numpy as np
import itertools
import scipy.linalg



def ket(dim, j):
    '''Input:
    dim: int or list, dimension of the ket
    j: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    '''
    def single_basis(j_val, d_val):
        vec = np.zeros(d_val, dtype=float)
        vec[j_val] = 1.0
        return vec

    if isinstance(j, int):
        if isinstance(dim, int):
            out = single_basis(j, dim)
        else:
            raise ValueError("If dim is a list, j must also be a list of the same length.")
    elif isinstance(j, list):
        if isinstance(dim, int):
            vectors = [single_basis(ji, dim) for ji in j]
        else:
            if len(j) != len(dim):
                raise ValueError("j and dim lists must have the same length.")
            vectors = [single_basis(ji, di) for ji, di in zip(j, dim)]
        out = vectors[0]
        for vec in vectors[1:]:
            out = np.kron(out, vec)
    else:
        raise TypeError("j must be an integer or a list of integers.")
    
    return out
