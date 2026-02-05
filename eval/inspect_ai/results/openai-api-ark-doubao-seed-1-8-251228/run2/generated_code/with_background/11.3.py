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


def multi_rail_encoding_state(rails):
    '''Returns the density matrix of the multi-rail encoding state
    Input:
    rails: int, number of rails
    Output:
    state: 2**(2*rails) x 2**(2*rails) dimensional array of numpy.float64 type
    '''
    m = rails
    # Generate all single-1 basis vectors for m qubits
    j_lists = []
    for k in range(m):
        j_list = [0] * m
        j_list[k] = 1
        j_lists.append(j_list)
    
    # Accumulate the sum of |s_i>⊗|s_i> terms
    psi = None
    for j_list in j_lists:
        combined_j = j_list + j_list
        term = ket([2] * (2 * m), combined_j)
        if psi is None:
            psi = term
        else:
            psi += term
    
    # Normalize the state vector
    psi /= np.sqrt(m)
    
    # Compute density matrix as outer product of the state with itself
    density_matrix = np.outer(psi, psi)
    
    return density_matrix



def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''
    processed = []
    for arg in args:
        if not isinstance(arg, np.ndarray):
            raise TypeError(f"All inputs must be numpy arrays. Received {type(arg)}.")
        if arg.ndim == 1:
            proc = arg.reshape(-1, 1)
        elif arg.ndim == 2:
            proc = arg
        else:
            raise ValueError(f"Input array has {arg.ndim} dimensions. Only 1D (vectors) and 2D (matrices) are allowed.")
        processed.append(proc)
    
    if not processed:
        return np.array([[1.0]], dtype=np.float64)
    
    result = processed[0]
    for mat in processed[1:]:
        result = np.kron(result, mat)
    
    return result
