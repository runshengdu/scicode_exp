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
    # Define spin-z operator in |↑⟩, |↓⟩ basis
    Sz1 = np.array([[0.5, 0.0],
                    [0.0, -0.5]], dtype=float)
    
    # Define spin ladder operator S^+ in |↑⟩, |↓⟩ basis
    Sp1 = np.array([[0.0, 1.0],
                    [0.0, 0.0]], dtype=float)
    
    # Single-site Hamiltonian is zero (no adjacent interaction partners)
    H1 = np.zeros((model_d, model_d), dtype=float)
    
    # Assemble operator dictionary
    operator_dict = {
        "H": H1,
        "conn_Sz": Sz1,
        "conn_Sp": Sp1
    }
    
    # Create and return the initial Block instance
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
    # Obtain spin minus operators via transpose (Hermitian conjugate for real matrices)
    Sm1 = Sp1.T
    Sm2 = Sp2.T
    
    # Calculate exchange interaction term (XX part)
    exchange_term = 0.5 * (kron(Sp1, Sm2) + kron(Sm1, Sp2))
    # Calculate Ising interaction term (Z part)
    z_term = kron(Sz1, Sz2)
    
    # Combine terms to form the two-site Hamiltonian
    H2_mat = exchange_term + z_term
    
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
    # Extract operators from the input block
    H_b = block.operator_dict["H"]
    S_r_z = block.operator_dict["conn_Sz"]
    S_r_p = block.operator_dict["conn_Sp"]
    
    # Define single-site operators for the new site
    Sz_d = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=float)
    Sp_d = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float)
    Sm_d = Sp_d.T  # S^- is the adjoint of S^+, real matrix so transpose
    H_d = np.zeros((model_d, model_d), dtype=float)  # Single-site Hamiltonian is zero
    
    # Convert block operators to sparse matrices
    H_b_sparse = sparse.csr_matrix(H_b)
    S_r_z_sparse = sparse.csr_matrix(S_r_z)
    S_r_p_sparse = sparse.csr_matrix(S_r_p)
    S_r_m_sparse = S_r_p_sparse.T  # S_r^- is transpose of S_r^+ (real matrices)
    I_b = sparse.identity(block.basis_size, format='csr')
    
    # Convert new site operators to sparse matrices
    Sz_d_sparse = sparse.csr_matrix(Sz_d)
    Sp_d_sparse = sparse.csr_matrix(Sp_d)
    Sm_d_sparse = sparse.csr_matrix(Sm_d)
    H_d_sparse = sparse.csr_matrix(H_d)
    I_d = sparse.identity(model_d, format='csr')
    
    # Compute each term of the enlarged Hamiltonian H_e
    term1 = kron(H_b_sparse, I_d)  # H_b ⊗ I_d
    term2 = kron(I_b, H_d_sparse)  # I_b ⊗ H_d (zero term)
    term3 = 0.5 * (kron(S_r_p_sparse, Sm_d_sparse) + kron(S_r_m_sparse, Sp_d_sparse))
    term4 = kron(S_r_z_sparse, Sz_d_sparse)
    
    H_e = term1 + term2 + term3 + term4
    
    # Compute connection operators for the enlarged block (act on the new rightmost site)
    conn_Sz_e = kron(I_b, Sz_d_sparse)
    conn_Sp_e = kron(I_b, Sp_d_sparse)
    
    # Prepare operator dictionary for the enlarged block
    operator_dict = {
        "H": H_e,
        "conn_Sz": conn_Sz_e,
        "conn_Sp": conn_Sp_e
    }
    
    # Calculate new block properties
    new_length = block.length + 1
    new_basis_size = block.basis_size * model_d
    
    # Create and return the EnlargedBlock instance
    eblock = EnlargedBlock(length=new_length, basis_size=new_basis_size, operator_dict=operator_dict)
    
    return eblock
