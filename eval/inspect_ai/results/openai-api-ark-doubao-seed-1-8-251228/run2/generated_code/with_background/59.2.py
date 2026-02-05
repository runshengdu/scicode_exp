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
    theta_half = theta / 2
    if axis == 1:
        c = np.cos(theta_half)
        s = np.sin(theta_half)
        R = np.array([[c, -1j * s],
                      [-1j * s, c]])
    elif axis == 2:
        c = np.cos(theta_half)
        s = np.sin(theta_half)
        R = np.array([[c, -s],
                      [s, c]])
    elif axis == 3:
        e1 = exp(-1j * theta_half)
        e2 = exp(1j * theta_half)
        R = np.array([[e1, 0],
                      [0, e2]])
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
    # Initial state |00⟩
    initial_state = np.array([[1], [0], [0], [0]], dtype=np.complex128)
    
    # Step: Generate |01⟩ by applying R_x(π) on the second qubit
    rx_pi = rotation_matrices(1, np.pi)
    h_operator = np.kron(np.eye(2), rx_pi)
    state = h_operator @ initial_state
    
    # Step 1: Apply R_y(π/2) on qubit 1 and R_x(-π/2) on qubit 2
    ry_pi_half = rotation_matrices(2, np.pi/2)
    rx_minus_pi_half = rotation_matrices(1, -np.pi/2)
    gate1 = np.kron(ry_pi_half, rx_minus_pi_half)
    state = gate1 @ state
    
    # Step 2: Apply CNOT gate with control on qubit 1 and target on qubit 2
    cnot12 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.complex128)
    state = cnot12 @ state
    
    # Step 3: Apply R_z(-2θ) on the second qubit
    rz_minus_2theta = rotation_matrices(3, -2 * theta)
    gate3 = np.kron(np.eye(2), rz_minus_2theta)
    state = gate3 @ state
    
    # Step 4: Apply CNOT12 again
    state = cnot12 @ state
    
    # Step 5: Apply R_y(-π/2) on qubit 1 and R_x(π/2) on qubit 2
    ry_minus_pi_half = rotation_matrices(2, -np.pi/2)
    rx_pi_half = rotation_matrices(1, np.pi/2)
    gate5 = np.kron(ry_minus_pi_half, rx_pi_half)
    state = gate5 @ state
    
    ansatz = state
    return ansatz
