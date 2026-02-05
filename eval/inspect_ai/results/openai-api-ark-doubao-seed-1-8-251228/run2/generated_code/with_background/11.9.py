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
    # Input validation for K
    if not isinstance(K, list):
        raise TypeError("K must be a list of 2D numpy arrays.")
    if len(K) == 0:
        raise ValueError("K must be a non-empty list of Kraus operators.")
    for k in K:
        if not isinstance(k, np.ndarray):
            raise TypeError(f"Each element in K must be a numpy array. Received {type(k)}.")
        if k.ndim != 2:
            raise ValueError(f"Each Kraus operator must be 2D. Got {k.ndim}D array.")
        if k.shape[0] != k.shape[1]:
            raise ValueError("All Kraus operators must be square matrices.")
    # Check all Kraus operators have consistent shape
    kraus_shape = K[0].shape
    for k in K[1:]:
        if k.shape != kraus_shape:
            raise ValueError("All Kraus operators must have identical dimensions.")
    
    # Input validation for rho
    if not isinstance(rho, np.ndarray):
        raise TypeError("rho must be a numpy array.")
    if rho.ndim != 2:
        raise ValueError(f"rho must be a 2D density matrix. Got {rho.ndim}D array.")
    if rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square matrix.")
    
    # Input validation for sys and dim
    if sys is not None:
        if dim is None:
            raise ValueError("dim must be provided when specifying sys as a list.")
        # Validate sys
        if not isinstance(sys, list):
            raise TypeError("sys must be a list of integers or None.")
        if not all(isinstance(s, int) for s in sys):
            raise TypeError("All elements in sys must be integers representing subsystem indices.")
        if len(sys) != len(set(sys)):
            raise ValueError("sys cannot contain duplicate subsystem indices.")
        # Validate dim
        if not isinstance(dim, list):
            raise TypeError("dim must be a list of integers or None.")
        if not all(isinstance(d, int) for d in dim):
            raise TypeError("All elements in dim must be integers representing subsystem dimensions.")
        if len(dim) == 0:
            raise ValueError("dim must be a non-empty list.")
        # Validate subsystem indices are within bounds
        max_subsys_idx = len(dim) - 1
        for s in sys:
            if not (0 <= s <= max_subsys_idx):
                raise ValueError(f"Subsystem index {s} is out of bounds (valid range: 0 to {max_subsys_idx}).")
        # Validate Kraus operators match subsystem dimensions
        kraus_dim = kraus_shape[0]
        for s in sys:
            if dim[s] != kraus_dim:
                raise ValueError(
                    f"Subsystem {s} has dimension {dim[s]}, but Kraus operators are {kraus_dim}-dimensional."
                )
        # Validate total dimension matches rho size
        total_dim = np.prod(dim)
        if total_dim != rho.shape[0]:
            raise ValueError(
                f"Total Hilbert space dimension from dim ({total_dim}) does not match rho size ({rho.shape[0]})."
            )
    else:
        if dim is not None:
            raise ValueError("dim must be None when sys is None (applying to full system).")
        # Validate Kraus operators match full system dimension
        for k in K:
            if k.shape != rho.shape:
                raise ValueError(
                    f"Kraus operator shape {k.shape} does not match rho shape {rho.shape} for full system application."
                )
    
    # Initialize result matrix
    result = np.zeros_like(rho, dtype=np.float64)
    
    if sys is None:
        # Apply channel to entire system directly
        for k in K:
            adjoint = k.conj().T
            result += k @ rho @ adjoint
    else:
        # Create mapping from subsystem index to its position in sys list
        sys_index_map = {s: idx for idx, s in enumerate(sys)}
        # Generate all combinations of Kraus operators (one per subsystem in sys)
        for kraus_tuple in itertools.product(K, repeat=len(sys)):
            # Build list of operators for each subsystem
            subsystem_ops = []
            for i in range(len(dim)):
                if i in sys_index_map:
                    # Use corresponding Kraus operator from the tuple
                    subsystem_ops.append(kraus_tuple[sys_index_map[i]])
                else:
                    # Identity operator for subsystems not in sys
                    subsystem_ops.append(np.eye(dim[i], dtype=np.float64))
            # Compute full Kraus operator via tensor product
            full_kraus = tensor(*subsystem_ops)
            # Calculate term and accumulate to result
            adjoint = full_kraus.conj().T
            result += full_kraus @ rho @ adjoint
    
    return result


def generalized_amplitude_damping_channel(gamma, N):
    '''Generates the generalized amplitude damping channel.
    Inputs:
    gamma: float, damping parameter
    N: float, thermal parameter
    Output:
    kraus: list of Kraus operators as 2x2 arrays of floats, [A1, A2, A3, A4]
    '''
    # Construct Kraus operator K1
    k1 = np.sqrt(1 - N) * np.array([[1.0, 0.0], [0.0, np.sqrt(1 - gamma)]])
    # Construct Kraus operator K2
    k2 = np.sqrt(gamma * (1 - N)) * np.array([[0.0, 1.0], [0.0, 0.0]])
    # Construct Kraus operator K3
    k3 = np.sqrt(N) * np.array([[np.sqrt(1 - gamma), 0.0], [0.0, 1.0]])
    # Construct Kraus operator K4
    k4 = np.sqrt(gamma * N) * np.array([[0.0, 0.0], [1.0, 0.0]])
    
    return [k1, k2, k3, k4]


def output_state(rails, gamma_1, N_1, gamma_2, N_2):
    '''Inputs:
    rails: int, number of rails
    gamma_1: float, damping parameter of the first channel
    N_1: float, thermal parameter of the first channel
    gamma_2: float, damping parameter of the second channel
    N_2: float, thermal parameter of the second channel
    Output
    state: 2**(2*rails) x 2**(2*rails) dimensional array of floats, the output state
    '''
    # Generate initial multi-rail encoded state
    initial_rho = multi_rail_encoding_state(rails)
    
    # Prepare for applying channels to receiver 1's rails
    kraus1 = generalized_amplitude_damping_channel(gamma_1, N_1)
    sys_receiver1 = list(range(rails))
    dim_list = [2] * (2 * rails)
    rho_after_receiver1 = apply_channel(kraus1, initial_rho, sys=sys_receiver1, dim=dim_list)
    
    # Prepare for applying channels to receiver 2's rails
    kraus2 = generalized_amplitude_damping_channel(gamma_2, N_2)
    sys_receiver2 = list(range(rails, 2 * rails))
    final_rho = apply_channel(kraus2, rho_after_receiver1, sys=sys_receiver2, dim=dim_list)
    
    return final_rho


def measurement(rails):
    '''Returns the measurement projector
    Input:
    rails: int, number of rails
    Output:
    global_proj: ( 2**(2*rails), 2**(2*rails) ) dimensional array of floats
    '''
    m = rails
    # Generate j_lists for one-particle sector of m qubits (exactly one 1 per list)
    j_lists = []
    for k in range(m):
        j_list = [0] * m
        j_list[k] = 1
        j_lists.append(j_list)
    
    # Compute projector P1 for single receiver's m qubits
    P1 = None
    for j_list in j_lists:
        vec = ket([2] * m, j_list)
        outer_product = np.outer(vec, vec)
        if P1 is None:
            P1 = outer_product
        else:
            P1 += outer_product
    
    # Compute global projector as tensor product of P1 with itself
    global_proj = tensor(P1, P1)
    
    return global_proj


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
    # Validate X
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if X.shape[0] != X.shape[1]:
        raise ValueError("X must be a square matrix.")
    
    # Validate dim
    if not isinstance(dim, list):
        raise TypeError("dim must be a list of integers.")
    if not all(isinstance(d, int) and d > 0 for d in dim):
        raise ValueError("All elements in dim must be positive integers.")
    n = len(dim)
    if n == 0:
        raise ValueError("dim must be a non-empty list.")
    total_dim = np.prod(dim)
    if total_dim != X.shape[0]:
        raise ValueError(
            f"Product of dimensions ({total_dim}) must match the dimension of X ({X.shape[0]})."
        )
    
    # Validate perm
    if not isinstance(perm, list):
        raise TypeError("perm must be a list of integers.")
    if not all(isinstance(p, int) for p in perm):
        raise TypeError("All elements in perm must be integers.")
    if len(perm) != n:
        raise ValueError(f"perm must have length {n}, matching the number of subsystems.")
    if sorted(perm) != list(range(n)):
        raise ValueError(f"perm must be a valid permutation of 0 to {n-1}.")
    
    # Reshape X into 2n-dimensional tensor
    X_tensor = X.reshape(dim + dim)
    
    # Compute permuted axes: permute ket and bra subsystems according to perm
    permuted_ket_axes = [perm[i] for i in range(n)]
    permuted_bra_axes = [n + perm[i] for i in range(n)]
    permuted_axes = permuted_ket_axes + permuted_bra_axes
    
    # Transpose the tensor to permute subsystems
    permuted_tensor = X_tensor.transpose(permuted_axes)
    
    # Reshape back to 2D matrix
    Y = permuted_tensor.reshape((total_dim, total_dim))
    
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
        raise TypeError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array. Got {X.ndim}D array.")
    if X.shape[0] != X.shape[1]:
        raise ValueError("X must be a square matrix.")
    
    # Input validation for dim
    if not isinstance(dim, list):
        raise TypeError("dim must be a list of integers.")
    if not all(isinstance(d, int) and d > 0 for d in dim):
        raise ValueError("All elements in dim must be positive integers.")
    total_dim = np.prod(dim)
    if total_dim != X.shape[0]:
        raise ValueError(
            f"Product of subsystem dimensions ({total_dim}) does not match X's dimension ({X.shape[0]})."
        )
    n = len(dim)
    
    # Input validation for sys
    if not isinstance(sys, list):
        raise TypeError("sys must be a list of integers.")
    if not all(isinstance(s, int) for s in sys):
        raise TypeError("All elements in sys must be integers.")
    if len(sys) != len(set(sys)):
        raise ValueError("sys cannot contain duplicate subsystem indices.")
    for s in sys:
        if not (0 <= s < n):
            raise ValueError(f"Subsystem index {s} is out of bounds (valid range: 0 to {n-1}).")
    
    # Determine kept subsystems and permutation to group kept first, traced last
    trace_subsystems = set(sys)
    keep_subsystems = sorted(set(range(n)) - trace_subsystems)
    perm = keep_subsystems + sorted(trace_subsystems)
    
    # Permute subsystems to rearrange traced subsystems to the end
    permuted_X = syspermute(X, perm, dim)
    
    # Calculate combined dimensions of kept and traced subsystems
    d_S = np.prod([dim[s] for s in keep_subsystems]) if keep_subsystems else 1
    d_T = np.prod([dim[t] for t in trace_subsystems]) if trace_subsystems else 1
    
    # Reshape permuted density matrix into 4D tensor for trace calculation
    permuted_X_reshaped = permuted_X.reshape(d_S, d_T, d_S, d_T)
    
    # Compute partial trace by summing over identical traced subsystem indices
    result = np.trace(permuted_X_reshaped, axis1=1, axis2=3)
    
    return result.astype(np.float64)
