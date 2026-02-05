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


def multi_rail_encoding_state(rails):
    '''Returns the density matrix of the multi-rail encoding state
    Input:
    rails: int, number of rails
    Output:
    state: 2**(2*rails) x 2**(2*rails) dimensional array of numpy.float64 type
    '''
    m = rails
    total_dim = 2 ** (2 * m)
    sum_ket = np.zeros(total_dim, dtype=np.float64)
    
    for i in range(m):
        # Create the j list for the m-qubit state with 1 at position i
        j_list = [0] * m
        j_list[i] = 1
        # Get the ket vector for the m-qubit state
        e_ket = ket(j_list, 2)
        # Compute the tensor product of the state with itself
        term = np.kron(e_ket, e_ket)
        # Add to the cumulative sum
        sum_ket += term
    
    # Normalize the state vector
    psi_ket = sum_ket / np.sqrt(m)
    # Compute the density matrix via outer product
    rho = np.outer(psi_ket, psi_ket)
    
    return rho



def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''
    if not args:
        return np.array([[1.0]], dtype=np.float64)
    
    processed = []
    for arg in args:
        arr = np.asarray(arg, dtype=np.float64)
        if arr.ndim == 1:
            processed_arr = arr.reshape(-1, 1)
        elif arr.ndim == 2:
            processed_arr = arr
        else:
            raise ValueError("All inputs must be 1D or 2D arrays.")
        processed.append(processed_arr)
    
    result = processed[0]
    for mat in processed[1:]:
        result = np.kron(result, mat)
    
    return result
