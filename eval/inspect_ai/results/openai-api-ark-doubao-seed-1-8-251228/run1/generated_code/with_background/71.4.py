import numpy as np 
from scipy.optimize import fminbound
import itertools
from scipy.linalg import logm


def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''
    if not args:
        return np.array([[1.0]])
    
    processed = []
    for arr in args:
        if arr.ndim == 1:
            # Reshape 1D array to column vector (2D) and convert to float
            processed_arr = arr.reshape(-1, 1).astype(float)
        elif arr.ndim == 2:
            # Convert to float
            processed_arr = arr.astype(float)
        else:
            raise ValueError("All input arrays must be 1D or 2D.")
        processed.append(processed_arr)
    
    # Compute iterative Kronecker product
    result = processed[0]
    for current in processed[1:]:
        result = np.kron(result, current)
    
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
    total_dim = rho.shape[0]
    assert rho.shape[1] == total_dim, "rho must be a square matrix"
    
    # Handle cases where sys or dim is None
    if sys is None:
        # Channel acts on entire system
        result = np.zeros_like(rho, dtype=float)
        for k in K:
            result += k @ rho @ k.conj().T
        return result
    
    if dim is None:
        # Entire system is one subsystem, sys must be [0]
        assert sys == [0], "sys must be [0] when dim is None"
        result = np.zeros_like(rho, dtype=float)
        for k in K:
            result += k @ rho @ k.conj().T
        return result
    
    # Now handle the general case where sys and dim are specified
    num_subsystems = len(dim)
    assert all(0 <= s < num_subsystems for s in sys), "Invalid subsystem index in sys"
    assert np.prod(dim) == total_dim, "Product of subsystem dimensions must equal rho dimension"
    
    all_subsystems = list(range(num_subsystems))
    complement = [s for s in all_subsystems if s not in sys]
    permuted_order = sys + complement  # Order: sys first, then complement
    dim_sys = [dim[s] for s in sys]
    dim_complement = [dim[c] for c in complement]
    dim_S = np.prod(dim_sys)
    dim_C = np.prod(dim_complement)
    
    # Step 1: Reshape rho into tensor form with separate subsystem indices
    rho_tensor = rho.reshape(dim + dim)
    
    # Step 2: Transpose to group sys subsystems first in both rows and columns
    transpose_order = permuted_order + [s + num_subsystems for s in permuted_order]
    rho_transposed = rho_tensor.transpose(transpose_order)
    
    # Step 3: Reshape to (dim_S, dim_C, dim_S, dim_C) for easier processing
    rho_reshaped = rho_transposed.reshape((dim_S, dim_C, dim_S, dim_C))
    
    # Step 4: Apply each Kraus operator and sum
    result_reshaped = np.zeros_like(rho_reshaped, dtype=float)
    for k in K:
        # Compute k @ rho_slice @ k.conj().T for all slices using einsum
        term = np.einsum('ab, bced, fd -> acfe', k, rho_reshaped, k)
        result_reshaped += term
    
    # Step 5: Reverse the permutation to get back to original order
    # Reshape back to permuted tensor form
    permuted_shape = tuple(dim_sys + dim_complement + dim_sys + dim_complement)
    result_tensor = result_reshaped.reshape(permuted_shape)
    
    # Compute inverse permutation to reverse the transpose
    inv_perm = [permuted_order.index(s) for s in all_subsystems]
    reverse_transpose_order = inv_perm + [s + num_subsystems for s in inv_perm]
    result_tensor = result_tensor.transpose(reverse_transpose_order)
    
    # Reshape back to matrix form
    matrix = result_tensor.reshape((total_dim, total_dim))
    
    return matrix



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
    assert X.shape[0] == X.shape[1], "X must be a square matrix"
    num_subsystems = len(dim)
    assert len(perm) == num_subsystems, "Permutation length must match number of subsystems"
    assert sorted(perm) == list(range(num_subsystems)), "Permutation must be a valid permutation of subsystem indices"
    assert np.prod(dim) == X.shape[0], "Product of subsystem dimensions must equal X's dimension"
    
    # Reshape X into tensor form with explicit subsystem dimensions for bra and ket
    x_tensor = X.reshape(dim + dim)
    
    # Define transpose order: permute ket indices first, then bra indices in the same permutation
    transpose_order = perm + [num_subsystems + s for s in perm]
    
    # Transpose tensor to reorder subsystems
    x_transposed = x_tensor.transpose(transpose_order)
    
    # Reshape back to matrix form
    Y = x_transposed.reshape(X.shape)
    
    return Y
