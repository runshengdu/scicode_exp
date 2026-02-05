import numpy as np
from cmath import exp
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.linalg import expm

def rotation_matrices(axis, theta):
    '''Create rotation matrices Rx, Ry, and Rz with the given angle theta.
    Inputs:
    axis : int
        The rotation axis. 1 = x, 2 = y, 3 = z.
    theta : float
        The rotation angle.
    Output:
    R : matrix of shape(2, 2)
        The rotation matrix.
    '''
    theta_over_2 = theta / 2
    if axis == 1:
        c = np.cos(theta_over_2)
        s = np.sin(theta_over_2)
        R = np.array([[c, -1j * s],
                      [-1j * s, c]])
    elif axis == 2:
        c = np.cos(theta_over_2)
        s = np.sin(theta_over_2)
        R = np.array([[c, -s],
                      [s, c]])
    elif axis == 3:
        diag1 = exp(-1j * theta_over_2)
        diag2 = exp(1j * theta_over_2)
        R = np.array([[diag1, 0],
                      [0, diag2]])
    else:
        raise ValueError("Axis must be 1 (x), 2 (y), or 3 (z).")
    return R


def create_ansatz(theta):
    '''Create the ansatz wavefunction with a given theta.
    Input:
    theta : float
        The only variational parameter.
    Output:
    ansatz : array of shape (4, 1)
        The ansatz wavefunction.
    '''
    # Initial state |00> as column vector
    initial_state = np.array([[1], [0], [0], [0]])
    
    # Step 0: Apply R_x(pi) to second qubit to get |01> (with phase)
    R_x_pi = rotation_matrices(1, np.pi)
    U0 = np.kron(np.eye(2), R_x_pi)
    state = U0 @ initial_state
    
    # Step 1: Apply R_y(pi/2) on qubit1 and R_x(-pi/2) on qubit2
    R_y_pi2 = rotation_matrices(2, np.pi/2)
    R_x_minus_pi2 = rotation_matrices(1, -np.pi/2)
    U1 = np.kron(R_y_pi2, R_x_minus_pi2)
    state = U1 @ state
    
    # Step 2: Apply CNOT12
    U2 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]])
    state = U2 @ state
    
    # Step3: Apply R_z(-2*theta) on qubit2
    R_z_minus_2theta = rotation_matrices(3, -2 * theta)
    U3 = np.kron(np.eye(2), R_z_minus_2theta)
    state = U3 @ state
    
    # Step4: Apply CNOT12 again
    U4 = U2
    state = U4 @ state
    
    # Step5: Apply R_y(-pi/2) on qubit1 and R_x(pi/2) on qubit2
    R_y_minus_pi2 = rotation_matrices(2, -np.pi/2)
    R_x_pi2 = rotation_matrices(1, np.pi/2)
    U5 = np.kron(R_y_minus_pi2, R_x_pi2)
    state = U5 @ state
    
    ansatz = state
    return ansatz


def measureZ(U, psi):
    '''Perform a measurement in the Z-basis for a 2-qubit system where only Pauli Sz measurements are possible.
    The measurement is applied to the first qubit.
    Inputs:
    U : matrix of shape(4, 4)
        The unitary transformation to be applied before measurement.
    psi : array of shape (4, 1)
        The two-qubit state before the unitary transformation.
    Output:
    measured_result: float
        The result of the Sz measurement after applying U.
    '''
    # Apply the unitary transformation to the state
    psi_prime = U @ psi
    # Define the Z1 operator (Pauli Z on first qubit, identity on second)
    Z = np.array([[1, 0], [0, -1]])
    Z1_op = np.kron(Z, np.eye(2))
    # Calculate the expectation value
    expectation = (psi_prime.conj().T @ Z1_op @ psi_prime).item()
    # Return the real part as the result (guaranteed real for Hermitian operators)
    return np.real(expectation)




def projective_expected(theta, gl):
    '''Calculate the expectation value of the energy with proper unitary transformations.
    Input:
    theta : float
        The only variational parameter.
    gl = [g0, g1, g2, g3, g4, g5] : array in size 6
        Hamiltonian coefficients.
    Output:
    energy : float
        The expectation value of the energy with the given parameter theta.
    '''
    # Get the ansatz wavefunction
    psi = create_ansatz(theta)
    
    # Unitary for Z1⊗I (g1 term)
    U1 = np.eye(4)
    
    # Unitary for I⊗Z2 (g2 term): SWAP matrix
    U2 = np.array([[1, 0, 0, 0],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1]])
    
    # Unitary for Z1⊗Z2 (g3 term): CNOT21 matrix (control qubit 2, target qubit 1)
    CNOT21 = np.array([[1, 0, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0]])
    U3 = CNOT21
    
    # Unitary for Y1⊗Y2 (g4 term): CNOT21 @ (HS†⊗HS†)
    H = (1 / np.sqrt(2)) * np.array([[1, 1],
                                     [1, -1]])
    S_dagger = np.array([[1, 0],
                         [0, -1j]])
    HS_dagger = H @ S_dagger
    HS_dagger_tensor = np.kron(HS_dagger, HS_dagger)
    U4 = CNOT21 @ HS_dagger_tensor
    
    # Unitary for X1⊗X2 (g5 term): CNOT21 @ (H⊗H)
    H_tensor = np.kron(H, H)
    U5 = CNOT21 @ H_tensor
    
    # Calculate expectation values using measureZ function
    exp1 = measureZ(U1, psi)
    exp2 = measureZ(U2, psi)
    exp3 = measureZ(U3, psi)
    exp4 = measureZ(U4, psi)
    exp5 = measureZ(U5, psi)
    
    # Compute total energy expectation value
    energy = (gl[0] + gl[1] * exp1 + gl[2] * exp2 +
              gl[3] * exp3 + gl[4] * exp4 + gl[5] * exp5)
    
    # Return real part to eliminate tiny imaginary components from numerical errors
    return np.real(energy)

# The following functions are required for completeness (as they depend on previous steps)
def rotation_matrices(axis, theta):
    '''Create rotation matrices Rx, Ry, and Rz with the given angle theta.
    Inputs:
    axis : int
        The rotation axis. 1 = x, 2 = y, 3 = z.
    theta : float
        The rotation angle.
    Output:
    R : matrix of shape(2, 2)
        The rotation matrix.
    '''
    theta_over_2 = theta / 2
    if axis == 1:
        c = np.cos(theta_over_2)
        s = np.sin(theta_over_2)
        R = np.array([[c, -1j * s],
                      [-1j * s, c]])
    elif axis == 2:
        c = np.cos(theta_over_2)
        s = np.sin(theta_over_2)
        R = np.array([[c, -s],
                      [s, c]])
    elif axis == 3:
        diag1 = np.exp(-1j * theta_over_2)
        diag2 = np.exp(1j * theta_over_2)
        R = np.array([[diag1, 0],
                      [0, diag2]])
    else:
        raise ValueError("Axis must be 1 (x), 2 (y), or 3 (z).")
    return R

def create_ansatz(theta):
    '''Create the ansatz wavefunction with a given theta.
    Input:
    theta : float
        The only variational parameter.
    Output:
    ansatz : array of shape (4, 1)
        The ansatz wavefunction.
    '''
    # Initial state |00> as column vector
    initial_state = np.array([[1], [0], [0], [0]])
    
    # Step 0: Apply R_x(pi) to second qubit to get |01> (with phase)
    R_x_pi = rotation_matrices(1, np.pi)
    U0 = np.kron(np.eye(2), R_x_pi)
    state = U0 @ initial_state
    
    # Step 1: Apply R_y(pi/2) on qubit1 and R_x(-pi/2) on qubit2
    R_y_pi2 = rotation_matrices(2, np.pi/2)
    R_x_minus_pi2 = rotation_matrices(1, -np.pi/2)
    U1 = np.kron(R_y_pi2, R_x_minus_pi2)
    state = U1 @ state
    
    # Step 2: Apply CNOT12
    U2 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]])
    state = U2 @ state
    
    # Step3: Apply R_z(-2*theta) on qubit2
    R_z_minus_2theta = rotation_matrices(3, -2 * theta)
    U3 = np.kron(np.eye(2), R_z_minus_2theta)
    state = U3 @ state
    
    # Step4: Apply CNOT12 again
    U4 = U2
    state = U4 @ state
    
    # Step5: Apply R_y(-pi/2) on qubit1 and R_x(pi/2) on qubit2
    R_y_minus_pi2 = rotation_matrices(2, -np.pi/2)
    R_x_pi2 = rotation_matrices(1, np.pi/2)
    U5 = np.kron(R_y_minus_pi2, R_x_pi2)
    state = U5 @ state
    
    ansatz = state
    return ansatz

def measureZ(U, psi):
    '''Perform a measurement in the Z-basis for a 2-qubit system where only Pauli Sz measurements are possible.
    The measurement is applied to the first qubit.
    Inputs:
    U : matrix of shape(4, 4)
        The unitary transformation to be applied before measurement.
    psi : array of shape (4, 1)
        The two-qubit state before the unitary transformation.
    Output:
    measured_result: float
        The result of the Sz measurement after applying U.
    '''
    # Apply the unitary transformation to the state
    psi_prime = U @ psi
    # Define the Z1 operator (Pauli Z on first qubit, identity on second)
    Z = np.array([[1, 0], [0, -1]])
    Z1_op = np.kron(Z, np.eye(2))
    # Calculate the expectation value
    expectation = (psi_prime.conj().T @ Z1_op @ psi_prime).item()
    # Return the real part as the result (guaranteed real for Hermitian operators)
    return np.real(expectation)
