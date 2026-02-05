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


def apply_channel(K, rho, sys=None, dim=None):
    '''Applies the channel with Kraus operators in K to the state rho on
    systems specified by the list sys. The dimensions of the subsystems of
    rho are given by dim.
    Inputs:
    K: list of 2d array of floats, list of Kraus operators
    rho: 2d array of floats, input density matrix
    sys: list of int or None, list of subsystems to apply the channel, None means full system
    dim: list of int or None, list of dimensions of each subsystem, None means full system
    Output:
    matrix: output density matrix of floats
    '''
    # Validate rho is a square 2D array
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square 2D array")
    
    # Validate K is a non-empty list of square 2D arrays
    if not isinstance(K, list) or len(K) == 0:
        raise ValueError("K must be a non-empty list of Kraus operators")
    d_k = None
    for k in K:
        if not isinstance(k, np.ndarray):
            raise TypeError("Each Kraus operator must be a numpy array")
        if k.ndim != 2 or k.shape[0] != k.shape[1]:
            raise ValueError("Each Kraus operator must be a square 2D array")
        if d_k is None:
            d_k = k.shape[0]
        else:
            if k.shape[0] != d_k:
                raise ValueError("All Kraus operators must have the same shape")
    
    if sys is None:
        # Case 1: Apply channel to the full system
        if dim is not None:
            raise ValueError("dim must be None when sys is None")
        D = rho.shape[0]
        if d_k != D:
            raise ValueError(f"Kraus operators are {d_k}x{d_k}, but rho is {D}x{D}")
        result = np.zeros_like(rho, dtype=np.float64)
        for k in K:
            result += k @ rho @ np.conj(k.T)
        return result
    else:
        # Case 2: Apply channel to specified subsystems
        # Validate sys is a list of integers
        if not isinstance(sys, list) or not all(isinstance(s, int) for s in sys):
            raise TypeError("sys must be a list of integers")
        # Validate dim is a list of positive integers
        if not isinstance(dim, list) or not all(isinstance(d, int) and d > 0 for d in dim):
            raise TypeError("dim must be a list of positive integers")
        n_subsys = len(dim)
        if n_subsys == 0:
            raise ValueError("dim must be a non-empty list")
        # Validate subsystem indices are within bounds
        for s in sys:
            if s < 0 or s >= n_subsys:
                raise ValueError(f"Subsystem index {s} is out of bounds (0 to {n_subsys-1})")
        # Validate rho's dimension matches product of subsystem dimensions
        D = np.prod(dim)
        if rho.shape[0] != D:
            raise ValueError(f"rho dimension {rho.shape[0]} does not match product of subsystem dimensions {D}")
        # Validate all subsystems in sys match Kraus operator dimension
        for s in sys:
            if dim[s] != d_k:
                raise ValueError(f"Subsystem {s} has dimension {dim[s]}, but Kraus operators are {d_k}x{d_k}")
        
        # Initialize result to zero matrix
        result = np.zeros_like(rho, dtype=np.float64)
        m = len(sys)
        # Iterate over all combinations of Kraus operators for each occurrence in sys
        for kraus_tuple in itertools.product(K, repeat=m):
            # Collect operators for each subsystem
            ops_per_subsys = [[] for _ in range(n_subsys)]
            for i in range(m):
                t = sys[i]
                ops_per_subsys[t].append(kraus_tuple[i])
            # Build the list of operators for tensor product
            operators = []
            for t in range(n_subsys):
                ops_t = ops_per_subsys[t]
                d_t = dim[t]
                if not ops_t:
                    # No operators for this subsystem: use identity
                    op = np.eye(d_t, dtype=np.float64)
                else:
                    # Compute product of operators in application order
                    op = ops_t[0]
                    for next_op in ops_t[1:]:
                        op = next_op @ op
                operators.append(op)
            # Compute full Kraus operator via tensor product
            full_kraus = tensor(*operators)
            # Compute term and add to result
            term = full_kraus @ rho @ np.conj(full_kraus.T)
            result += term
        
        return result


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



def generalized_amplitude_damping_channel(gamma, N):
    '''Generates the generalized amplitude damping channel.
    Inputs:
    gamma: float, damping parameter
    N: float, thermal parameter
    Output:
    kraus: list of Kraus operators as 2x2 arrays of floats, [A1, A2, A3, A4]
    '''
    # Compute diagonal Kraus operators
    K1 = np.sqrt(1 - N) * np.diag([1.0, np.sqrt(1 - gamma)])
    K3 = np.sqrt(N) * np.diag([np.sqrt(1 - gamma), 1.0])
    
    # Compute off-diagonal Kraus operators
    K2 = np.sqrt(gamma * (1 - N)) * np.array([[0.0, 1.0], [0.0, 0.0]])
    K4 = np.sqrt(gamma * N) * np.array([[0.0, 0.0], [1.0, 0.0]])
    
    return [K1, K2, K3, K4]
