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
    # Define single-site spin operators in 2x2 matrix form
    Sz = np.array([[0.5, 0.0],
                   [0.0, -0.5]], dtype=float)
    Sp = np.array([[0.0, 1.0],
                   [0.0, 0.0]], dtype=float)
    
    # Single-site Hamiltonian is zero (no neighboring sites to interact with)
    H1 = np.zeros((model_d, model_d), dtype=float)
    
    # Construct operator dictionary as required
    operator_dict = {
        "H": H1,
        "conn_Sz": Sz,
        "conn_Sp": Sp
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
    # Spin minus operator is the transpose of spin plus operator (for real matrices)
    Sp1_minus = Sp1.T
    Sp2_minus = Sp2.T
    
    # Calculate interaction terms
    exchange_term = 0.5 * (kron(Sp1, Sp2_minus) + kron(Sp1_minus, Sp2))
    sz_coupling_term = kron(Sz1, Sz2)
    
    # Total two-site Hamiltonian
    H2_mat = exchange_term + sz_coupling_term
    
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


    # Extract parameters from the input block
    m = block.basis_size
    original_length = block.length

    # Extract operators from the block and convert to sparse matrices
    H_b = block.operator_dict["H"]
    S_r_z = block.operator_dict["conn_Sz"]
    S_r_p = block.operator_dict["conn_Sp"]

    H_b_sparse = csr_matrix(H_b)
    S_r_z_sparse = csr_matrix(S_r_z)
    S_r_p_sparse = csr_matrix(S_r_p)
    S_r_m_sparse = S_r_p_sparse.T  # Spin minus is transpose of spin plus for real matrices

    # Define new single-site operators as sparse matrices
    H_d_sparse = csr_matrix(np.zeros((model_d, model_d), dtype=float))
    S_z_d_sparse = csr_matrix([[0.5, 0.0],
                               [0.0, -0.5]], dtype=float)
    S_p_d_sparse = csr_matrix([[0.0, 1.0],
                               [0.0, 0.0]], dtype=float)
    S_m_d_sparse = S_p_d_sparse.T

    # Identity matrices for block and new site
    I_b = identity(m, format='csr')
    I_d = identity(model_d, format='csr')

    # Calculate Hamiltonian terms for the enlarged block
    term1 = kron(H_b_sparse, I_d)  # Block Hamiltonian extended to enlarged system
    term2 = kron(I_b, H_d_sparse)  # New site Hamiltonian (zero for single site)
    term3 = 0.5 * (kron(S_r_p_sparse, S_m_d_sparse) + kron(S_r_m_sparse, S_p_d_sparse))  # Exchange interaction
    term4 = kron(S_r_z_sparse, S_z_d_sparse)  # Spin-z coupling

    # Total Hamiltonian of the enlarged system
    H_e = term1 + term2 + term3 + term4

    # Calculate connection operators for the enlarged block (spin operators on new site)
    S_e_z = kron(I_b, S_z_d_sparse)
    S_e_p = kron(I_b, S_p_d_sparse)

    # Construct operator dictionary for the enlarged block
    operator_dict = {
        "H": H_e,
        "conn_Sz": S_e_z,
        "conn_Sp": S_e_p
    }

    # Calculate new block parameters
    new_length = original_length + 1
    new_basis_size = m * model_d

    # Create and return the EnlargedBlock instance
    eblock = EnlargedBlock(length=new_length, basis_size=new_basis_size, operator_dict=operator_dict)

    return eblock
