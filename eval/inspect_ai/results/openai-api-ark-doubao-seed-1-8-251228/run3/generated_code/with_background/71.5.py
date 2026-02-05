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
    # Validate rho is square
    if rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square matrix")
    
    # Validate K is list of 2D square arrays
    if not isinstance(K, list) or any(not isinstance(k, np.ndarray) or k.ndim != 2 or k.shape[0] != k.shape[1] for k in K):
        raise ValueError("K must be a list of 2D square numpy arrays")
    
    d_full = rho.shape[0]
    
    # Case 1: Channel acts on entire system (sys is None)
    if sys is None:
        # Check all Kraus operators are full dimension
        for k in K:
            if k.shape != (d_full, d_full):
                raise ValueError(f"All Kraus operators must be {d_full}x{d_full} when sys is None")
        # Compute channel action
        result = np.zeros_like(rho, dtype=float)
        for k in K:
            result += k @ rho @ k.conj().T
        return result
    
    # Case 2: Channel acts on specified subsystems (sys is not None)
    # Validate dim is provided and valid
    if dim is None:
        raise ValueError("dim must be provided when sys is not None")
    if not isinstance(dim, list) or any(not isinstance(d, int) or d <= 0 for d in dim):
        raise ValueError("dim must be a list of positive integers")
    
    # Helper function to compute product of a list
    def product(lst):
        p = 1
        for x in lst:
            p *= x
        return p
    
    # Check product of dim equals full dimension
    if product(dim) != d_full:
        raise ValueError(f"Product of dimensions in dim ({product(dim)}) must equal the dimension of rho ({d_full})")
    
    # Validate sys is valid
    if not isinstance(sys, list) or any(not isinstance(s, int) for s in sys):
        raise ValueError("sys must be a list of integers")
    n_subsys = len(dim)
    for s in sys:
        if s < 0 or s >= n_subsys:
            raise ValueError(f"Subsystem index {s} is out of bounds (0 to {n_subsys - 1})")
    if len(set(sys)) != len(sys):
        raise ValueError("sys must contain unique subsystem indices")
    
    # Check Kraus operators have correct dimension for the subsystems
    D_sys = product(dim[s] for s in sys)
    for k in K:
        if k.shape != (D_sys, D_sys):
            raise ValueError(f"All Kraus operators must be {D_sys}x{D_sys} for subsystems {sys}")
    
    # Prepare permutation of subsystems to bring sys to front
    all_subsys = list(range(n_subsys))
    non_sys = [s for s in all_subsys if s not in sys]
    perm = sys + non_sys
    perm_dim = [dim[s] for s in perm]
    
    # Helper function to compute index of a tuple in the tensor product basis
    def get_index(tup, dim_list):
        idx = 0
        mult = 1
        # Iterate from last subsystem to first (little-endian order)
        for i in reversed(range(len(dim_list))):
            idx += tup[i] * mult
            mult *= dim_list[i]
        return idx
    
    # Construct permutation matrix P
    P = np.zeros((d_full, d_full), dtype=float)
    # Generate all possible index tuples for the original system
    all_tuples = list(itertools.product(*[range(d) for d in dim]))
    for t in all_tuples:
        old_idx = get_index(t, dim)
        # Permute the tuple to match the permuted subsystem order
        t_perm = tuple(t[s] for s in perm)
        new_idx = get_index(t_perm, perm_dim)
        P[new_idx, old_idx] = 1.0
    
    # Compute identity on non-sys subsystems
    D_non_sys = product(dim[s] for s in non_sys)
    identity_non_sys = np.eye(D_non_sys, dtype=float)
    
    # Compute channel action
    result = np.zeros_like(rho, dtype=float)
    for k in K:
        # Construct Kraus operator on permuted system: k ⊗ identity_non_sys
        k_perm = tensor(k, identity_non_sys)
        # Transform back to original system to get full Kraus operator
        k_full = P.T @ k_perm @ P
        # Add the contribution to the result
        result += k_full @ rho @ k_full.conj().T
    
    return result



def syspermute(X, perm, dim):
    '''Permutes order of subsystems in the multipartite operator X.
    Inputs:
    X: 2d array of floats with equal dimensions, the density matrix of the state
    perm: list of int containing the desired order
    dim: list of int containing the dimensions of all subsystems.
    Output:
    Y: 2d array of floats with equal dimensions, the density matrix of the permuted state
    '''
    # Input validations
    if X.shape[0] != X.shape[1]:
        raise ValueError("X must be a square matrix")
    
    if not isinstance(dim, list) or any(not isinstance(d, int) or d <= 0 for d in dim):
        raise ValueError("dim must be a list of positive integers")
    
    # Compute total dimension
    D = 1
    for d in dim:
        D *= d
    
    if D != X.shape[0]:
        raise ValueError(f"Product of dimensions in dim ({D}) must equal the dimension of X ({X.shape[0]})")
    
    if not isinstance(perm, list) or any(not isinstance(s, int) for s in perm):
        raise ValueError("perm must be a list of integers")
    
    if len(perm) != len(dim):
        raise ValueError("perm must have the same length as dim")
    
    if sorted(perm) != list(range(len(dim))):
        raise ValueError(f"perm must be a permutation of 0 to {len(dim)-1}.")
    
    n = len(dim)
    
    # Reshape X into 2n-dimensional array where first n axes are bra indices, last n are ket indices
    X_reshaped = X.reshape(dim + dim)
    
    # Create axis permutation: permute both bra and ket axes according to the given permutation
    bra_perm = perm
    ket_perm = [n + s for s in perm]
    new_axes = bra_perm + ket_perm
    
    # Transpose the array to permute the subsystems
    X_transposed = X_reshaped.transpose(new_axes)
    
    # Reshape back to a D x D matrix
    Y = X_transposed.reshape((D, D))
    
    return Y



def partial_trace(X, sys, dim):
    '''Inputs:
    X: 2d array of floats with equal dimensions, the density matrix of the state
    sys: list of int containing systems over which to take the partial trace (i.e., the systems to discard).
    dim: list of int containing dimensions of all subsystems.
    Output:
    2d array of floats with equal dimensions, density matrix after partial trace.
    '''
    # Input validations
    if X.shape[0] != X.shape[1]:
        raise ValueError("X must be a square matrix")
    
    if not isinstance(dim, list):
        raise TypeError("dim must be a list of positive integers")
    for d in dim:
        if not isinstance(d, int) or d <= 0:
            raise ValueError("All elements in dim must be positive integers")
    
    D = np.prod(dim)
    if D != X.shape[0]:
        raise ValueError(f"Product of dimensions in dim ({D}) must equal the dimension of X ({X.shape[0]})")
    
    if not isinstance(sys, list):
        raise TypeError("sys must be a list of integers")
    for s in sys:
        if not isinstance(s, int):
            raise TypeError("All elements in sys must be integers")
    
    n_subsys = len(dim)
    for s in sys:
        if s < 0 or s >= n_subsys:
            raise ValueError(f"Subsystem index {s} is out of bounds (0 to {n_subsys - 1})")
    
    if len(set(sys)) != len(sys):
        raise ValueError("sys must contain unique subsystem indices")
    
    # Determine keep subsystems and permutation to group keepers first
    keep_sys = [s for s in range(n_subsys) if s not in sys]
    perm = keep_sys + sys
    
    # Permute the state to rearrange subsystems
    X_perm = syspermute(X, perm, dim)
    
    # Calculate dimensions of the kept and traced-out subsystems
    D_keep = np.prod([dim[s] for s in keep_sys])
    D_trace = np.prod([dim[s] for s in sys])
    
    # Reshape permuted state into 4D tensor: (keep_row, trace_row, keep_col, trace_col)
    X_reshaped = X_perm.reshape(D_keep, D_trace, D_keep, D_trace)
    
    # Compute partial trace by summing over matching trace indices in rows and columns
    result = np.trace(X_reshaped, axis1=1, axis2=3)
    
    return result
