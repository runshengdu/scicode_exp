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
    # Generate initial m-rail encoded bipartite state
    rho_initial = multi_rail_encoding_state(rails)
    
    # Get Kraus operators for receiver 1's channels
    kraus1 = generalized_amplitude_damping_channel(gamma_1, N_1)
    # Define subsystems for receiver 1 (first 'rails' qubits) and full system dimensions
    sys_receiver1 = list(range(rails))
    full_dim = [2] * (2 * rails)
    # Apply channels to receiver 1's qubits
    rho_after_receiver1 = apply_channel(kraus1, rho_initial, sys=sys_receiver1, dim=full_dim)
    
    # Get Kraus operators for receiver 2's channels
    kraus2 = generalized_amplitude_damping_channel(gamma_2, N_2)
    # Define subsystems for receiver 2 (last 'rails' qubits)
    sys_receiver2 = list(range(rails, 2 * rails))
    # Apply channels to receiver 2's qubits
    rho_final = apply_channel(kraus2, rho_after_receiver1, sys=sys_receiver2, dim=full_dim)
    
    return rho_final


def measurement(rails):
    '''Returns the measurement projector
    Input:
    rails: int, number of rails
    Output:
    global_proj: ( 2**(2*rails), 2**(2*rails) ) dimensional array of floats
    '''
    m = rails
    size_p = 2 ** m
    P1 = np.zeros((size_p, size_p), dtype=np.float64)
    
    for i in range(m):
        j_list = [0] * m
        j_list[i] = 1
        e_ket = ket(j_list, 2)
        P1 += np.outer(e_ket, e_ket)
    
    global_proj = np.kron(P1, P1)
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
    # Validate X is a square 2D array
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError("X must be a square 2D array")
    
    # Validate perm is a list of integers
    if not isinstance(perm, list) or not all(isinstance(p, int) for p in perm):
        raise TypeError("perm must be a list of integers")
    
    # Validate dim is a list of positive integers
    if not isinstance(dim, list) or not all(isinstance(d, int) and d > 0 for d in dim):
        raise TypeError("dim must be a list of positive integers")
    
    # Validate length of perm matches length of dim
    n = len(dim)
    if len(perm) != n:
        raise ValueError(f"Length of perm ({len(perm)}) must match length of dim ({n})")
    
    # Validate perm is a valid permutation of 0..n-1
    if sorted(perm) != list(range(n)):
        raise ValueError(f"perm must be a permutation of 0 to {n-1}")
    
    # Validate product of dim matches X's size
    D = np.prod(dim)
    if X.shape[0] != D:
        raise ValueError(f"Product of dimensions ({D}) does not match size of X ({X.shape[0]})")
    
    # Reshape X into multi-dimensional tensor (ket indices followed by bra indices)
    tensor_reshaped = X.reshape(*dim, *dim)
    
    # Create new axes order: permute both ket and bra subsystems
    ket_axes = perm
    bra_axes = [n + p for p in perm]
    new_axes = ket_axes + bra_axes
    
    # Transpose tensor to permute subsystems
    tensor_permuted = tensor_reshaped.transpose(new_axes)
    
    # Reshape back to 2D density matrix
    Y = tensor_permuted.reshape(D, D)
    
    return Y


def partial_trace(X, sys, dim):
    '''Inputs:
    X: 2d array of floats with equal dimensions, the density matrix of the state
    sys: list of int containing systems over which to take the partial trace (i.e., the systems to discard).
    dim: list of int containing dimensions of all subsystems.
    Output:
    2d array of floats with equal dimensions, density matrix after partial trace.
    '''
    # Validate X is a square 2D array
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError("X must be a square 2D array")
    
    # Validate dim is a list of positive integers
    if not isinstance(dim, list) or not all(isinstance(d, int) and d > 0 for d in dim):
        raise TypeError("dim must be a list of positive integers")
    n_subsys = len(dim)
    if n_subsys == 0:
        raise ValueError("dim cannot be an empty list")
    
    # Validate product of dimensions matches X's size
    total_dim = np.prod(dim)
    if X.shape[0] != total_dim:
        raise ValueError(f"Product of subsystem dimensions ({total_dim}) does not match size of X ({X.shape[0]})")
    
    # Validate sys is a list of unique, valid integers
    if not isinstance(sys, list) or not all(isinstance(s, int) for s in sys):
        raise TypeError("sys must be a list of integers")
    if len(set(sys)) != len(sys):
        raise ValueError("sys contains duplicate subsystem indices")
    for s in sys:
        if s < 0 or s >= n_subsys:
            raise ValueError(f"Subsystem index {s} is out of bounds (0 to {n_subsys-1})")
    
    # Determine kept subsystems (in original order) and permutation
    traced_set = set(sys)
    kept_sys = [s for s in range(n_subsys) if s not in traced_set]
    perm = kept_sys + sys  # Permute: kept subsystems first, then traced
    
    # Permute the subsystems to group kept and traced
    Y = syspermute(X, perm, dim)
    
    # Calculate dimensions of kept and traced subspaces
    D_kept = int(np.prod([dim[s] for s in kept_sys])) if kept_sys else 1
    D_traced = int(np.prod([dim[s] for s in sys])) if sys else 1
    
    # Reshape Y to separate kept and traced indices
    Y_reshaped = Y.reshape(D_kept, D_traced, D_kept, D_traced)
    
    # Compute partial trace by summing over traced indices where ket and bra are equal
    rho_kept = np.einsum('ikjk->ij', Y_reshaped)
    
    return rho_kept



def entropy(rho):
    '''Inputs:
    rho: 2d array of floats with equal dimensions, the density matrix of the state
    Output:
    en: quantum (von Neumann) entropy of the state rho, float
    '''
    # Validate rho is a square 2D array
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square 2D array")
    
    # Compute eigenvalues of the Hermitian density matrix
    eigenvalues = np.linalg.eigvalsh(rho)
    
    # Calculate entropy terms, handling zero/negative eigenvalues safely
    entropy_terms = np.where(eigenvalues > 1e-12, eigenvalues * np.log2(eigenvalues), 0.0)
    en = -np.sum(entropy_terms)
    
    return en


def coherent_inf_state(rho_AB, dimA, dimB):
    '''Inputs:
    rho_AB: 2d array of floats with equal dimensions, the state we evaluate coherent information
    dimA: int, dimension of system A
    dimB: int, dimension of system B
    Output
    co_inf: float, the coherent information of the state rho_AB
    '''
    # Compute von Neumann entropy of the full bipartite state
    S_AB = entropy(rho_AB)
    
    # Compute reduced state of subsystem B by tracing out subsystem A
    rho_B = partial_trace(rho_AB, sys=[0], dim=[dimA, dimB])
    
    # Compute von Neumann entropy of the reduced state of B
    S_B = entropy(rho_B)
    
    # Calculate coherent information using the formula I_c = S(B) - S(AB)
    co_inf = S_B - S_AB
    
    return co_inf



def rate(rails, gamma_1, N_1, gamma_2, N_2):
    '''Inputs:
    rails: int, number of rails
    gamma_1: float, damping parameter of the first channel
    N_1: float, thermal parameter of the first channel
    gamma_2: float, damping parameter of the second channel
    N_2: float, thermal parameter of the second channel
    Output: float, the achievable rate of our protocol
    '''
    # Get the output state after passing through both channels
    rho_output = output_state(rails, gamma_1, N_1, gamma_2, N_2)
    
    # Get the global measurement projector for the one-particle sector
    proj = measurement(rails)
    
    # Compute unnormalized post-measurement state
    rho_unnormalized = proj @ rho_output @ proj
    
    # Calculate measurement success probability
    p = np.trace(rho_unnormalized)
    
    # Handle case where measurement probability is effectively zero
    if p < 1e-15:
        return 0.0
    
    # Normalize to get the post-measurement density matrix
    rho_prime = rho_unnormalized / p
    
    # Calculate dimension of each receiver's Hilbert space
    dim_per_receiver = 2 ** rails
    
    # Compute coherent information of the post-measurement state
    coh_info = coherent_inf_state(rho_prime, dim_per_receiver, dim_per_receiver)
    
    # Calculate the rate and ensure non-negative achievable rate
    rate_val = (p * coh_info) / rails
    return max(rate_val, 0.0)
