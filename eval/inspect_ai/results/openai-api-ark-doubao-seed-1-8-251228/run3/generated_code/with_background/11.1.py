import numpy as np
import itertools
import scipy.linalg




def ket(j, dim):
    '''Input:
    j: int or list, the index(es) of the basis vector(s)
    dim: int or list, dimension(s) of the ket space(s)
    Output:
    out: numpy array of float, the matrix representation of the ket tensor product
    '''
    # Validate input types and values
    def validate_single(j_i, d_i):
        if not isinstance(j_i, int) or j_i < 0 or j_i >= d_i:
            raise ValueError(f"Index {j_i} is invalid for dimension {d_i}")
        vec = np.zeros(d_i, dtype=float)
        vec[j_i] = 1.0
        return vec

    if isinstance(dim, int):
        if isinstance(j, int):
            return validate_single(j, dim)
        elif isinstance(j, list):
            # Compute tensor product of d-dimensional basis vectors
            vectors = [validate_single(ji, dim) for ji in j]
            result = vectors[0]
            for vec in vectors[1:]:
                result = np.kron(result, vec)
            return result
        else:
            raise TypeError("j must be int or list when dim is int")
    elif isinstance(dim, list):
        if not isinstance(j, list):
            raise TypeError("j must be list when dim is list")
        if len(j) != len(dim):
            raise ValueError("Length of j must match length of dim")
        # Compute tensor product of mixed-dimensional basis vectors
        vectors = [validate_single(ji, di) for ji, di in zip(j, dim)]
        result = vectors[0]
        for vec in vectors[1:]:
            result = np.kron(result, vec)
        return result
    else:
        raise TypeError("dim must be int or list of positive integers")
