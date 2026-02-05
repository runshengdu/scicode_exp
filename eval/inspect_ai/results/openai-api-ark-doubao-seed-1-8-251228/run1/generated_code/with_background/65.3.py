import numpy as np
from scipy.linalg import sqrtm
import itertools

def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''
    if not args:
        raise ValueError("At least one matrix or vector must be provided.")
    
    processed_arrays = []
    for arg in args:
        arr = np.asarray(arg)
        if arr.ndim == 1:
            # Convert 1D vector to 2D column vector
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            raise ValueError("All inputs must be 1D (vectors) or 2D (matrices).")
        processed_arrays.append(arr)
    
    # Compute successive Kronecker products
    result = processed_arrays[0]
    for mat in processed_arrays[1:]:
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
    # Convert rho to numpy array and validate
    rho = np.asarray(rho)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square 2D array (density matrix).")
    
    # Validate Kraus operators
    if not isinstance(K, list) or len(K) == 0:
        raise ValueError("K must be a non-empty list of Kraus operators.")
    k_shapes = set()
    for k in K:
        k_arr = np.asarray(k)
        if k_arr.ndim != 2 or k_arr.shape[0] != k_arr.shape[1]:
            raise ValueError(f"Kraus operator {k} must be a square 2D matrix.")
        k_shapes.add(k_arr.shape)
    if len(k_shapes) != 1:
        raise ValueError("All Kraus operators must have the same shape.")
    k_dim = k_shapes.pop()[0]
    
    # Handle different cases for sys and dim
    if sys is None and dim is None:
        # Apply channel to entire system
        d = rho.shape[0]
        if k_dim != d:
            raise ValueError(f"Kraus operators are {k_dim}x{k_dim}, but rho is {d}x{d}.")
        result = np.zeros_like(rho)
        for k in K:
            k_arr = np.asarray(k)
            result += k_arr @ rho @ k_arr.conj().T
        return result
    elif isinstance(sys, list) and isinstance(dim, list):
        # Apply channel to specified subsystems
        # Validate dim
        if len(dim) == 0:
            raise ValueError("dim must be a non-empty list of subsystem dimensions.")
        for d in dim:
            if not isinstance(d, int) or d <= 0:
                raise ValueError(f"Subsystem dimension {d} must be a positive integer.")
        total_dim = np.prod(dim)
        if rho.shape[0] != total_dim:
            raise ValueError(f"rho is {rho.shape[0]}x{rho.shape[0]}, but product of dim is {total_dim}.")
        
        # Validate sys
        sys_set = set()
        for s in sys:
            if not isinstance(s, int):
                raise ValueError(f"Subsystem index {s} must be an integer.")
            if s < 0 or s >= len(dim):
                raise ValueError(f"Subsystem index {s} is out of range (0 to {len(dim)-1}).")
            if s in sys_set:
                raise ValueError(f"Subsystem index {s} is duplicated in sys.")
            sys_set.add(s)
            if dim[s] != k_dim:
                raise ValueError(f"Subsystem {s} has dimension {dim[s]}, but Kraus operators are {k_dim}x{k_dim}.")
        
        # Precompute identity matrices for each subsystem
        identities = [np.eye(d, dtype=rho.dtype) for d in dim]
        # Map subsystem indices to their positions in the sys list
        sys_to_tuple_idx = {s: i for i, s in enumerate(sys)}
        
        result = np.zeros_like(rho)
        # Iterate over all combinations of Kraus operators for target subsystems
        for tuple_k in itertools.product(K, repeat=len(sys)):
            op_list = []
            for s in range(len(dim)):
                if s in sys_to_tuple_idx:
                    # Get the corresponding Kraus operator from the tuple
                    k = tuple_k[sys_to_tuple_idx[s]]
                    k_arr = np.asarray(k)
                    op_list.append(k_arr)
                else:
                    # Use identity matrix for non-target subsystems
                    op_list.append(identities[s])
            # Compute combined Kraus operator using tensor product
            combined_k = tensor(*op_list)
            # Apply the combined Kraus operator to rho
            result += combined_k @ rho @ combined_k.conj().T
        return result
    else:
        # Invalid combination of sys and dim
        raise ValueError("Either both sys and dim are None, or sys is a list and dim is a list.")



def channel_output(input_state, channel1, channel2=None):
    '''Returns the channel output
    Inputs:
        input_state: density matrix of the input 2n qubit state, ( 2**(2n), 2**(2n) ) array of floats
        channel1: kruas operators of the first channel, list of (2,2) array of floats
        channel2: kruas operators of the second channel, list of (2,2) array of floats
    Output:
        output: the channel output, ( 2**(2n), 2**(2n) ) array of floats
    '''
    # Handle default case for channel2
    if channel2 is None:
        channel2 = channel1
    
    # Calculate total number of qubits and validate input state dimension
    dim_rho = input_state.shape[0]
    total_qubits = np.log2(dim_rho)
    if not total_qubits.is_integer():
        raise ValueError("input_state dimension is not a power of 2, invalid qubit state.")
    total_qubits = int(total_qubits)
    if total_qubits % 2 != 0:
        raise ValueError("input_state is not a 2n-qubit state (total qubits count is odd).")
    n = total_qubits // 2
    
    # Define dimension list for each qubit subsystem
    dim_list = [2] * total_qubits
    
    # Apply channel1 to the first n qubits
    first_subsystems = list(range(n))
    intermediate_state = apply_channel(channel1, input_state, sys=first_subsystems, dim=dim_list)
    
    # Apply channel2 to the last n qubits
    last_subsystems = list(range(n, total_qubits))
    output = apply_channel(channel2, intermediate_state, sys=last_subsystems, dim=dim_list)
    
    return output
