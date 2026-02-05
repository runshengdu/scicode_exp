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




def syspermute(X, perm, dim):
    '''Permutes order of subsystems in the multipartite operator X.
    Inputs:
    X: 2d array of floats with equal dimensions, the density matrix of the state
    perm: list of int containing the desired order
    dim: list of int containing the dimensions of all subsystems.
    Output:
    Y: 2d array of floats with equal dimensions, the density matrix of the permuted state
    '''
    # Input validation
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy ndarray")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X.shape[0] != X.shape[1]:
        raise ValueError("X must be a square matrix")
    
    n = len(dim)
    if len(perm) != n:
        raise ValueError(f"Length of perm ({len(perm)}) must match length of dim ({n})")
    if sorted(perm) != list(range(n)):
        raise ValueError(f"perm must be a valid permutation of 0 to {n-1} (inclusive)")
    
    total_dim = np.prod(dim)
    if X.shape != (total_dim, total_dim):
        raise ValueError(f"X's shape {X.shape} does not match total dimension {total_dim} from dim")
    
    # Convert to float to ensure output type correctness
    X_float = X.astype(float)
    
    # Reshape X into 2n-dimensional array (bra indices followed by ket indices)
    X_reshaped = X_float.reshape(dim + dim)
    
    # Create new axes order: permuted bra indices followed by permuted ket indices
    new_axes = perm + [n + p for p in perm]
    
    # Permute the axes to get the reshaped permuted state
    Y_reshaped = X_reshaped.transpose(new_axes)
    
    # Reshape back to 2D matrix
    Y = Y_reshaped.reshape(total_dim, total_dim)
    
    return Y



def partial_trace(X, sys, dim):
    '''Inputs:
    X: 2d array of floats with equal dimensions, the density matrix of the state
    sys: list of int containing systems over which to take the partial trace (i.e., the systems to discard).
    dim: list of int containing dimensions of all subsystems.
    Output:
    2d array of floats with equal dimensions, density matrix after partial trace.
    '''
    # Input validation for X
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy ndarray")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X.shape[0] != X.shape[1]:
        raise ValueError("X must be a square matrix")
    
    # Input validation for dim
    if not isinstance(dim, list):
        raise TypeError("dim must be a list of positive integers")
    for d in dim:
        if not isinstance(d, int) or d <= 0:
            raise ValueError(f"dim must contain positive integers, found {d}")
    n_subsystems = len(dim)
    
    # Input validation for sys
    if not isinstance(sys, list):
        raise TypeError("sys must be a list of integers")
    sys_set = set(sys)
    for s in sys:
        if not isinstance(s, int) or s < 0 or s >= n_subsystems:
            raise ValueError(f"Invalid subsystem index {s}: must be between 0 and {n_subsystems-1}")
    if len(sys_set) != len(sys):
        raise ValueError("sys contains duplicate subsystem indices")
    
    # Check X shape matches total dimension from dim
    total_dim = np.prod(dim, dtype=int)
    if X.shape != (total_dim, total_dim):
        raise ValueError(f"X's shape {X.shape} does not match total dimension {total_dim} from dim")
    
    # Convert X to float to ensure consistent output type
    X_float = X.astype(float)
    
    # Separate subsystems into keep and trace groups, preserving original order
    keep_sys = [s for s in range(n_subsystems) if s not in sys_set]
    trace_sys = [s for s in range(n_subsystems) if s in sys_set]
    
    # Create permutation to group keep subsystems first, then trace subsystems
    perm = keep_sys + trace_sys
    
    # Permute the state to group target subsystems
    X_perm = syspermute(X_float, perm, dim)
    
    # Calculate dimensions of keep and trace subsystems
    d_K = np.prod([dim[s] for s in keep_sys], dtype=int)
    d_T = np.prod([dim[s] for s in trace_sys], dtype=int)
    
    # Reshape permuted state into 4D array: (d_K, d_T, d_K, d_T)
    X_perm_reshaped = X_perm.reshape(d_K, d_T, d_K, d_T)
    
    # Compute partial trace by summing over matching trace indices in bra and ket
    partial_trace_result = np.einsum('acbc->ab', X_perm_reshaped)
    
    return partial_trace_result
