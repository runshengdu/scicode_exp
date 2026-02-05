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
    
    # Process each input to ensure it's 2D
    processed_mats = []
    for mat in args:
        if mat.ndim == 1:
            # Reshape 1D vector to 2D column vector
            processed = mat.reshape(-1, 1)
        elif mat.ndim == 2:
            processed = mat
        else:
            raise ValueError("All inputs must be 1D (vectors) or 2D (matrices).")
        processed_mats.append(processed)
    
    # Compute iterative Kronecker product
    result = processed_mats[0]
    for mat in processed_mats[1:]:
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
    # Validate sys and dim compatibility
    if sys is None and dim is not None:
        raise ValueError("dim must be None when sys is None.")
    if sys is not None and dim is None:
        raise ValueError("dim must be provided when sys is not None.")

    # Check rho is a valid 2D square matrix
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a 2D square density matrix.")

    # Case 1: Channel acts on the entire system
    if sys is None and dim is None:
        d = rho.shape[0]
        # Validate Kraus operators match system dimension
        for k in K:
            if k.ndim != 2 or k.shape != (d, d):
                raise ValueError(f"All Kraus operators must be {d}x{d} square matrices for full-system application.")
        # Compute channel action
        result = np.zeros_like(rho, dtype=np.complex128)
        for k in K:
            result += k @ rho @ k.conj().T
        return result.astype(float)

    # Case 2: Channel acts on specified subsystems
    # Validate sys and dim are lists
    if not isinstance(sys, list) or not isinstance(dim, list):
        raise ValueError("sys and dim must be lists when targeting specific subsystems.")
    
    n_subsystems = len(dim)
    # Validate subsystem indices are valid
    for s in sys:
        if not isinstance(s, int) or s < 0 or s >= n_subsystems:
            raise ValueError(f"Invalid subsystem index {s}. Must be between 0 and {n_subsystems-1}.")
    
    # Check total system dimension matches rho size
    total_dim = np.prod(dim)
    if rho.shape != (total_dim, total_dim):
        raise ValueError(f"rho must be {total_dim}x{total_dim} for subsystem dimensions {dim}.")
    
    # Validate Kraus operators are non-empty, square, and uniform size
    if not K:
        raise ValueError("Kraus operator list K cannot be empty.")
    kraus_dim = K[0].shape[0]
    if K[0].shape[0] != K[0].shape[1]:
        raise ValueError("All Kraus operators must be square matrices.")
    for k in K:
        if k.shape != (kraus_dim, kraus_dim):
            raise ValueError("All Kraus operators must have identical dimensions.")
    
    # Validate Kraus operators match target subsystem dimensions
    for s in sys:
        if dim[s] != kraus_dim:
            raise ValueError(f"Kraus operators are {kraus_dim}x{kraus_dim}, but subsystem {s} has dimension {dim[s]}.")
    
    # Compute channel action by iterating over all Kraus combinations
    result = np.zeros_like(rho, dtype=np.complex128)
    for kraus_comb in itertools.product(K, repeat=len(sys)):
        # Build components for full Kraus operator tensor product
        tensor_components = []
        for sub_idx in range(n_subsystems):
            if sub_idx in sys:
                # Get corresponding Kraus operator from the combination
                comb_pos = sys.index(sub_idx)
                tensor_components.append(kraus_comb[comb_pos])
            else:
                # Identity matrix for unaffected subsystems
                tensor_components.append(np.eye(dim[sub_idx]))
        # Construct full Kraus operator using tensor function
        full_kraus = tensor(*tensor_components)
        # Accumulate channel transformation
        result += full_kraus @ rho @ full_kraus.conj().T
    
    # Convert back to float type since inputs are real
    return result.astype(float)


def channel_output(input_state, channel1, channel2=None):
    '''Returns the channel output
    Inputs:
        input_state: density matrix of the input 2n qubit state, ( 2**(2n), 2**(2n) ) array of floats
        channel1: kruas operators of the first channel, list of (2,2) array of floats
        channel2: kruas operators of the second channel, list of (2,2) array of floats
    Output:
        output: the channel output, ( 2**(2n), 2**(2n) ) array of floats
    '''
    # Handle default channel2
    if channel2 is None:
        channel2 = channel1
    
    # Validate input_state is square
    if input_state.shape[0] != input_state.shape[1]:
        raise ValueError("input_state must be a square matrix.")
    
    # Calculate total number of qubits and n
    total_dim = input_state.shape[0]
    # Check if total dimension is a power of two
    if (total_dim & (total_dim - 1)) != 0:
        raise ValueError("Input state dimension must be a power of two (2^(2n)).")
    total_qubits = int(np.log2(total_dim))
    n = total_qubits // 2
    
    # Ensure total_qubits is even (2n qubits)
    if 2 * n != total_qubits:
        raise ValueError("Input state must correspond to an even number of qubits (2n qubits).")
    
    # Define subsystem dimensions: each qubit is a 2-dimensional subsystem
    dim = [2] * total_qubits
    
    # Apply channel1 to the first n qubits (subsystems 0 to n-1)
    intermediate_state = apply_channel(channel1, input_state, sys=list(range(n)), dim=dim)
    
    # Apply channel2 to the last n qubits (subsystems n to 2n-1)
    output_state = apply_channel(channel2, intermediate_state, sys=list(range(n, total_qubits)), dim=dim)
    
    return output_state



def ghz_protocol(state):
    '''Returns the output state of the protocol
    Input:
    state: 2n qubit input state, 2^2n by 2^2n array of floats, where n is determined from the size of the input state
    Output:
    post_selected: the output state
    '''
    # Validate input state is square
    if state.shape[0] != state.shape[1]:
        raise ValueError("Input state must be a square matrix.")
    
    total_dim = state.shape[0]
    # Check if total dimension is a power of two
    if (total_dim & (total_dim - 1)) != 0:
        raise ValueError("Input state dimension must be a power of two (2^(2n)).")
    
    total_qubits = int(np.log2(total_dim))
    # Verify we have an even number of qubits (2n)
    if total_qubits % 2 != 0:
        raise ValueError("Input state must correspond to an even number of qubits (2n qubits).")
    
    n = total_qubits // 2
    size_n = 2 ** n
    
    # Construct n-qubit projector onto |0^n> and |1^n>
    P_n = np.zeros((size_n, size_n), dtype=float)
    P_n[0, 0] = 1.0
    P_n[-1, -1] = 1.0
    
    # Construct total projector for 2n-qubit state: P_n ⊗ P_n
    P = tensor(P_n, P_n)
    
    # Compute state projected onto the parity measurement subspace
    projected = P @ state @ P
    
    # Calculate trace for normalization
    trace_val = np.trace(projected)
    
    # Handle case where there's no overlap with the projector
    if trace_val <= 0:
        return np.zeros((4, 4), dtype=float)
    
    # Normalize to get the post-measurement state ρ'
    rho_prime = projected / trace_val
    
    # Construct partial isometry for n qubits: maps |0^n>→|0>, |1^n>→|1>
    V_n = np.zeros((2, size_n), dtype=float)
    V_n[0, 0] = 1.0
    V_n[1, size_n - 1] = 1.0
    
    # Construct total isometry for 2n-qubit state: V_n ⊗ V_n
    V_total = tensor(V_n, V_n)
    
    # Apply partial isometry transformation to get final state
    post_selected = V_total @ rho_prime @ V_total.T
    
    return post_selected
