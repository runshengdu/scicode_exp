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
    
    # Convert 1D inputs to 2D column vectors and validate input dimensions
    processed = []
    for arg in args:
        if arg.ndim == 1:
            processed.append(arg.reshape(-1, 1))
        elif arg.ndim == 2:
            processed.append(arg)
        else:
            raise ValueError("All inputs must be 1D vectors or 2D matrices.")
    
    # Compute iterative Kronecker product leveraging associativity
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
    # Validate rho is a square 2D matrix
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square 2D matrix.")
    
    # Validate Kraus operators
    if not isinstance(K, list) or len(K) == 0:
        raise ValueError("K must be a non-empty list of Kraus operators.")
    k_dim = None
    for ki in K:
        if ki.ndim != 2 or ki.shape[0] != ki.shape[1]:
            raise ValueError("All Kraus operators must be square 2D matrices.")
        if k_dim is None:
            k_dim = ki.shape[0]
        else:
            if ki.shape[0] != k_dim:
                raise ValueError("All Kraus operators must have the same dimensions.")
    
    # Validate sys and dim consistency
    if (sys is None) != (dim is None):
        raise ValueError("sys and dim must both be None or both be lists.")
    
    # Case 1: Channel acts on entire system
    if sys is None and dim is None:
        if k_dim != rho.shape[0]:
            raise ValueError("Kraus operators must match the dimension of rho when acting on the entire system.")
        result = np.zeros_like(rho, dtype=complex)
        for ki in K:
            result += ki @ rho @ ki.conj().T
        return result
    
    # Case 2: Channel acts on specified subsystems
    else:
        # Validate sys is a list of integers
        if not isinstance(sys, list) or not all(isinstance(s, int) for s in sys):
            raise ValueError("sys must be a list of integers.")
        # Validate dim is a list of positive integers
        if not isinstance(dim, list) or not all(isinstance(d, int) and d > 0 for d in dim):
            raise ValueError("dim must be a list of positive integers.")
        
        # Check subsystem dimensions product matches rho's dimension
        total_dim = np.prod(dim)
        if total_dim != rho.shape[0]:
            raise ValueError(f"Product of subsystem dimensions ({total_dim}) does not match rho's dimension ({rho.shape[0]}).")
        
        # Validate subsystem indices are within range
        num_subsystems = len(dim)
        for s in sys:
            if s < 0 or s >= num_subsystems:
                raise ValueError(f"Subsystem index {s} is out of range. Must be between 0 and {num_subsystems - 1}.")
        
        # Validate target subsystems match Kraus operator dimension
        for s in sys:
            if dim[s] != k_dim:
                raise ValueError(f"Subsystem {s} has dimension {dim[s]}, but Kraus operators are {k_dim}x{k_dim}.")
        
        # Precompute identity matrices for each subsystem
        id_list = [np.eye(d, dtype=rho.dtype) for d in dim]
        
        # Generate all combinations of Kraus operators for each target subsystem
        kraus_tuples = itertools.product(K, repeat=len(sys))
        
        # Initialize result matrix
        result = np.zeros_like(rho, dtype=complex)
        
        # Iterate over all Kraus operator combinations
        for tpl in kraus_tuples:
            op_list = []
            for subsystem_idx in range(num_subsystems):
                # Collect all Kraus operators acting on current subsystem
                relevant_kraus = [tpl[i] for i in range(len(sys)) if sys[i] == subsystem_idx]
                
                if not relevant_kraus:
                    # No channel action: use identity
                    op = id_list[subsystem_idx]
                else:
                    # Compute product of Kraus operators in reverse application order
                    reversed_ops = reversed(relevant_kraus)
                    op = next(reversed_ops)
                    for mat in reversed_ops:
                        op = op @ mat
                
                op_list.append(op)
            
            # Compute full operator via tensor product
            full_op = tensor(*op_list)
            # Add the contribution of this Kraus operator combination
            result += full_op @ rho @ full_op.conj().T
        
        return result


def channel_output(input_state, channel1, channel2=None):
    '''Returns the channel output
    Inputs:
        input_state: density matrix of the input 2n qubit state, ( 2**(2n), 2**(2n) ) array of floats
        channel1: kruas operators of the first channel, list of (2,2) array of floats
        channel2: kruas operators of the second channel, list of (2,2) array of floats
    Output:
        output: the channel output, ( 2**(2n), 2**(2n) ) array of floats
    '''
    # Set channel2 to channel1 if not provided
    if channel2 is None:
        channel2 = channel1
    
    # Calculate total number of qubits and n
    total_qubits = int(np.log2(input_state.shape[0]))
    n = total_qubits // 2
    
    # Define dimension list for each qubit subsystem
    dim = [2] * total_qubits
    
    # Apply channel1 to the first n qubits
    first_subsystems = list(range(n))
    intermediate = apply_channel(channel1, input_state, sys=first_subsystems, dim=dim)
    
    # Apply channel2 to the last n qubits
    last_subsystems = list(range(n, total_qubits))
    final_state = apply_channel(channel2, intermediate, sys=last_subsystems, dim=dim)
    
    # Return real-valued result since all inputs are real
    return final_state.real


def ghz_protocol(state):
    '''Returns the output state of the protocol
    Input:
    state: 2n qubit input state, 2^2n by 2^2n array of floats, where n is determined from the size of the input state
    Output:
    post_selected: the output state
    '''
    # Validate input state is square
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("Input state must be a square 2D matrix.")
    
    # Calculate total number of qubits and n
    total_qubits = int(np.log2(state.shape[0]))
    if 2 ** total_qubits != state.shape[0]:
        raise ValueError("Input state must have size 2^(2n) for some integer n.")
    if total_qubits % 2 != 0:
        raise ValueError("Input state must be for an even number of qubits (2n qubits).")
    n = total_qubits // 2
    
    size_n = 2 ** n
    total_size = state.shape[0]
    
    # Define indices of basis states where both subsystems have even parity
    S = [
        0,  # |00...0> ⊗ |00...0>
        size_n - 1,  # |00...0> ⊗ |11...1>
        (size_n - 1) * size_n,  # |11...1> ⊗ |00...0>
        total_size - 1  # |11...1> ⊗ |11...1>
    ]
    
    # Extract the submatrix corresponding to valid parity combinations
    state_sub = state[S][:, S]
    
    # Calculate the probability of successful post-selection
    prob = np.trace(state_sub)
    
    # Handle zero probability case to avoid division by zero
    if prob == 0:
        post_selected = np.zeros((4, 4), dtype=state.dtype)
    else:
        # Normalize the submatrix to get the post-selected two-qubit state
        post_selected = state_sub / prob
    
    return post_selected



def fidelity(rho, sigma):
    '''Returns the fidelity between two states.
    Inputs:
    rho, sigma: density matrices of the two states, 2d array of floats
    Output:
    fid: fidelity, float
    '''
    # Validate rho is a square 2D matrix
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square 2D matrix.")
    # Validate sigma is a square 2D matrix
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("sigma must be a square 2D matrix.")
    # Validate rho and sigma have the same dimensions
    if rho.shape != sigma.shape:
        raise ValueError("rho and sigma must have the same dimensions.")
    
    # Compute matrix square roots of the density matrices
    sqrt_rho = sqrtm(rho)
    sqrt_sigma = sqrtm(sigma)
    
    # Compute the product of the square roots
    product = sqrt_rho @ sqrt_sigma
    
    # Compute the trace norm (sum of singular values) of the product
    singular_values = np.linalg.svd(product, compute_uv=False)
    trace_norm = np.sum(singular_values)
    
    # Calculate fidelity as the square of the trace norm, ensuring real value
    fid = np.real(trace_norm) ** 2
    
    # Convert to Python float and return
    return float(fid)
