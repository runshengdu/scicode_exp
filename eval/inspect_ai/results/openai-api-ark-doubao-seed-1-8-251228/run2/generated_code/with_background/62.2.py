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
