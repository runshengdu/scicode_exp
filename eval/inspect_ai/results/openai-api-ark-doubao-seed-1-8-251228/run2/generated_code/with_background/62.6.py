import numpy as np
from scipy.sparse import kron, identity
from scipy.sparse.linalg import eigsh  # Lanczos routine from ARPACK
def __init__(self, length, basis_size, operator_dict):
    self.length = length
    self.basis_size = basis_size
    self.operator_dict = operator_dict

def block_initial(model_d):
    '''Construct the initial block for the DMRG algo. H1, Sz1 and Sp1 is single-site Hamiltonian, spin-z operator
    and spin ladder operator in the form of 2x2 matrix, respectively.
    Input:
    model_d: int, single-site basis size
    Output:
    initial_block: instance of the "Block" class, with attributes "length", "basis_size", "operator_dict"
                  - length: An integer representing the block's current length.
                  - basis_size: An integer indicating the size of the basis.
                  - operator_dict: A dictionary containing operators: Hamiltonian ("H":H1), 
                                   Connection operator ("conn_Sz":Sz1), and Connection operator("conn_Sp":Sp1).
                                   H1, Sz1 and Sp1: 2d array of float
    '''
    # Define single-site spin operators as 2x2 matrices
    Sz1 = np.array([[0.5, 0.0], [0.0, -0.5]])
    Sp1 = np.array([[0.0, 1.0], [0.0, 0.0]])
    # Single-site Hamiltonian is zero (no neighboring sites)
    H1 = np.zeros((model_d, model_d))
    
    # Create operator dictionary with required keys
    operator_dict = {
        "H": H1,
        "conn_Sz": Sz1,
        "conn_Sp": Sp1
    }
    
    # Initialize and return the Block instance
    initial_block = Block(length=1, basis_size=model_d, operator_dict=operator_dict)
    
    return initial_block


def H_XXZ(Sz1, Sp1, Sz2, Sp2):
    '''Constructs the two-site Heisenberg XXZ chain Hamiltonian in matrix form.
    Input:
    Sz1,Sz2: 2d array of float, spin-z operator on site 1(or 2)
    Sp1,Sp2: 2d array of float, spin ladder operator on site 1(or 2)
    Output:
    H2_mat: sparse matrix of float, two-site Heisenberg XXZ chain Hamiltonian
    '''
    # Spin minus operator is the transpose of spin plus (for real matrices)
    Sm1 = Sp1.T
    Sm2 = Sp2.T
    
    # Calculate exchange interaction term: 1/2 (S₁⁺S₂⁻ + S₁⁻S₂⁺)
    exchange_term = 0.5 * (kron(Sp1, Sm2) + kron(Sm1, Sp2))
    
    # Calculate spin-z interaction term: S₁^z S₂^z
    sz_interaction = kron(Sz1, Sz2)
    
    # Total two-site Hamiltonian
    H2_mat = exchange_term + sz_interaction
    
    return H2_mat


def block_enlarged(block, model_d):
    '''Enlarges the given quantum block by one unit and updates its operators.
    Input:
    - block: instance of the "Block" class with the following attributes:
      - length: An integer representing the block's current length.
      - basis_size: An integer representing the size of the basis associated with the block.
      - operator_dict: A dictionary of quantum operators for the block:
          - "H": The Hamiltonian of the block.
          - "conn_Sz": A connection matrix, if length is 1, it corresponds to the spin-z operator.
          - "conn_Sp": A connection matrix, if length is 1, it corresponds to the spin ladder operator.
    - model_d: int, single-site basis size
    Output:
    - eblock: instance of the "EnlargedBlock" class with the following attributes:
      - length: An integer representing the new length.
      - basis_size: An integer representing the new size of the basis.
      - operator_dict: A dictionary of updated quantum operators:
          - "H": An updated Hamiltonian matrix of the enlarged system.
          - "conn_Sz": A new connection matrix.
          - "conn_Sp": Another new connection matrix.
          They are all sparse matrix
    '''
    # Convert block operators to sparse matrices
    H_b = csr_matrix(block.operator_dict["H"])
    S_r_z = csr_matrix(block.operator_dict["conn_Sz"])
    S_r_p = csr_matrix(block.operator_dict["conn_Sp"])
    S_r_m = S_r_p.T  # Spin minus is adjoint of spin plus (real matrices = transpose)
    
    # Define single-site operators (sparse matrices)
    S_z_d = np.array([[0.5, 0.0], [0.0, -0.5]])
    S_p_d = np.array([[0.0, 1.0], [0.0, 0.0]])
    S_m_d = S_p_d.T
    H_d = np.zeros((model_d, model_d))  # Single-site Hamiltonian is zero
    
    S_z_d_sparse = csr_matrix(S_z_d)
    S_p_d_sparse = csr_matrix(S_p_d)
    S_m_d_sparse = csr_matrix(S_m_d)
    H_d_sparse = csr_matrix(H_d)
    
    # Identity matrices for block and new site
    I_b = identity(block.basis_size)
    I_d = identity(model_d)
    
    # Calculate Hamiltonian terms for enlarged system
    term1 = kron(H_b, I_d)  # H_b ⊗ I_d
    term2 = kron(I_b, H_d_sparse)  # I_b ⊗ H_d (zero term, included for completeness)
    
    # Calculate interaction terms
    exchange_term = 0.5 * (kron(S_r_p, S_m_d_sparse) + kron(S_r_m, S_p_d_sparse))
    sz_interaction = kron(S_r_z, S_z_d_sparse)
    
    # Total Hamiltonian of enlarged block
    H_e = term1 + term2 + exchange_term + sz_interaction
    
    # Calculate connection operators for enlarged block (act on the new final site)
    conn_Sz_e = kron(I_b, S_z_d_sparse)
    conn_Sp_e = kron(I_b, S_p_d_sparse)
    
    # Build operator dictionary for enlarged block
    operator_dict_eblock = {
        "H": H_e,
        "conn_Sz": conn_Sz_e,
        "conn_Sp": conn_Sp_e
    }
    
    # Calculate new block properties
    new_length = block.length + 1
    new_basis_size = block.basis_size * model_d
    
    # Create and return EnlargedBlock instance
    eblock = EnlargedBlock(
        length=new_length,
        basis_size=new_basis_size,
        operator_dict=operator_dict_eblock
    )
    
    return eblock


def dmrg_module(sys, env, m, model_d):
    '''Input:
    sys: instance of the "Block" class
    env: instance of the "Block" class
    m: int, number of states in the new basis, i.e. the dimension of the new basis
    model_d: int, single-site basis size
    Output:
    newblock: instance of the "Block" class
    energy: superblock ground state energy, float
    '''
    # Step 1: Enlarge the system and environment blocks
    sys_enlarged = block_enlarged(sys, model_d)
    env_enlarged = block_enlarged(env, model_d)
    
    # Step 2: Extract operators from enlarged blocks
    # System enlarged operators
    H_sys_e = sys_enlarged.operator_dict["H"]
    conn_Sz_sys = sys_enlarged.operator_dict["conn_Sz"]
    conn_Sp_sys = sys_enlarged.operator_dict["conn_Sp"]
    conn_Sm_sys = conn_Sp_sys.T  # S^- is transpose of S^+ for real matrices
    
    # Environment enlarged operators
    H_env_e = env_enlarged.operator_dict["H"]
    conn_Sz_env = env_enlarged.operator_dict["conn_Sz"]
    conn_Sp_env = env_enlarged.operator_dict["conn_Sp"]
    conn_Sm_env = conn_Sp_env.T
    
    # Step 3: Build identity operators for the enlarged blocks
    I_sys_e = identity(sys_enlarged.basis_size)
    I_env_e = identity(env_enlarged.basis_size)
    
    # Step 4: Construct superblock Hamiltonian H_univ
    # Hamiltonian terms from individual blocks
    term1 = kron(H_sys_e, I_env_e)
    term2 = kron(I_sys_e, H_env_e)
    
    # Interaction terms between the two enlarged blocks
    exchange_term = 0.5 * (kron(conn_Sp_sys, conn_Sm_env) + kron(conn_Sm_sys, conn_Sp_env))
    sz_interaction = kron(conn_Sz_sys, conn_Sz_env)
    
    # Total superblock Hamiltonian
    H_univ = term1 + term2 + exchange_term + sz_interaction
    
    # Step 5: Compute ground state of H_univ with reproducible initial vector
    np.random.seed(42)
    dim_univ = sys_enlarged.basis_size * env_enlarged.basis_size
    v0 = np.random.rand(dim_univ)  # Fixed initial vector for reproducibility
    
    # Solve for the smallest eigenvalue (ground state) using eigsh
    eigenvalues, eigenvectors = eigsh(H_univ, k=1, which='SA', v0=v0)
    ground_energy = eigenvalues[0]
    ground_state = eigenvectors[:, 0]
    
    # Step 6: Compute reduced density matrix of the system
    # Reshape ground state into matrix form (sys_dim x env_dim)
    psi_matrix = ground_state.reshape(
        (sys_enlarged.basis_size, env_enlarged.basis_size),
        order='C'
    )
    # Reduced density matrix: rho_sys = Tr_env(|psi><psi|)
    rho_sys = psi_matrix @ psi_matrix.conj().T
    
    # Step 7: Get transformation matrix O from top m_tilde eigenvectors of rho_sys
    # Compute eigenvalues and eigenvectors of rho_sys
    vals_rho, vecs_rho = np.linalg.eigh(rho_sys)
    
    # Sort eigenvalues in descending order and get corresponding eigenvectors
    sorted_indices = np.argsort(vals_rho)[::-1]
    sorted_vecs_rho = vecs_rho[:, sorted_indices]
    
    # Determine number of states to keep
    m_tilde = min(m, env.basis_size * model_d)
    m_tilde = min(m_tilde, sys_enlarged.basis_size)  # Ensure we don't exceed available eigenvectors
    
    # Construct transformation matrix O
    O = sorted_vecs_rho[:, :m_tilde]
    
    # Step 8: Transform the operators of the enlarged system to get new block operators
    # Transform Hamiltonian
    H_sys_e_O = H_sys_e @ O
    new_H = O.conj().T @ H_sys_e_O
    
    # Transform spin-z connection operator
    conn_Sz_O = conn_Sz_sys @ O
    new_conn_Sz = O.conj().T @ conn_Sz_O
    
    # Transform spin-plus connection operator
    conn_Sp_O = conn_Sp_sys @ O
    new_conn_Sp = O.conj().T @ conn_Sp_O
    
    # Step 9: Create new Block instance
    operator_dict_new = {
        "H": new_H,
        "conn_Sz": new_conn_Sz,
        "conn_Sp": new_conn_Sp
    }
    newblock = Block(
        length=sys_enlarged.length,
        basis_size=m_tilde,
        operator_dict=operator_dict_new
    )
    
    return newblock, ground_energy



def run_dmrg(initial_block, L, m, model_d):
    '''Performs the Density Matrix Renormalization Group (DMRG) algorithm to find the ground state energy of a system.
    Input:
    - initial_block:an instance of the "Block" class with the following attributes:
        - length: An integer representing the current length of the block.
        - basis_size: An integer indicating the size of the basis.
        - operator_dict: A dictionary containing operators:
          Hamiltonian ("H"), Connection operator ("conn_Sz"), Connection operator("conn_Sp")
    - L (int): The desired system size (total length including the system and the environment).
    - m (int): The truncated dimension of the Hilbert space for eigenstate reduction.
    - model_d(int): Single-site basis size
    Output:
    - energy (float): The ground state energy of the infinite system after the DMRG steps.
    '''
    # Initialize system and environment blocks
    sys_block = initial_block
    env_block = initial_block
    
    total_length = sys_block.length + env_block.length
    if total_length >= L:
        # Compute ground energy of the initial superblock (no enlargement needed)
        # Convert operators to sparse matrices
        H_sys = csr_matrix(sys_block.operator_dict["H"])
        conn_Sz_sys = csr_matrix(sys_block.operator_dict["conn_Sz"])
        conn_Sp_sys = csr_matrix(sys_block.operator_dict["conn_Sp"])
        conn_Sm_sys = conn_Sp_sys.T
        
        H_env = csr_matrix(env_block.operator_dict["H"])
        conn_Sz_env = csr_matrix(env_block.operator_dict["conn_Sz"])
        conn_Sp_env = csr_matrix(env_block.operator_dict["conn_Sp"])
        conn_Sm_env = conn_Sp_env.T
        
        # Build identity matrices
        I_sys = identity(sys_block.basis_size)
        I_env = identity(env_block.basis_size)
        
        # Construct superblock Hamiltonian
        term1 = kron(H_sys, I_env)
        term2 = kron(I_sys, H_env)
        exchange_term = 0.5 * (kron(conn_Sp_sys, conn_Sm_env) + kron(conn_Sm_sys, conn_Sp_env))
        sz_interaction = kron(conn_Sz_sys, conn_Sz_env)
        H_univ = term1 + term2 + exchange_term + sz_interaction
        
        # Compute ground state energy
        np.random.seed(42)
        dim_univ = sys_block.basis_size * env_block.basis_size
        v0 = np.random.rand(dim_univ)
        eigenvalues, _ = eigsh(H_univ, k=1, which='SA', v0=v0)
        return eigenvalues[0]
    
    # Iterate DMRG steps until target length is reached or exceeded
    while True:
        new_block, current_energy = dmrg_module(sys_block, env_block, m, model_d)
        sys_block = new_block
        env_block = new_block
        
        total_length = sys_block.length + env_block.length
        if total_length >= L:
            return current_energy
