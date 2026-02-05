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
