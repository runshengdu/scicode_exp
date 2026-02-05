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
        cos_h = np.cos(theta_half)
        sin_h = np.sin(theta_half)
        R = np.array([[cos_h, -1j * sin_h],
                      [-1j * sin_h, cos_h]])
    elif axis == 2:
        cos_h = np.cos(theta_half)
        sin_h = np.sin(theta_half)
        R = np.array([[cos_h, -sin_h],
                      [sin_h, cos_h]])
    elif axis == 3:
        R = np.array([[exp(-1j * theta_half), 0],
                      [0, exp(1j * theta_half)]])
    else:
        raise ValueError("Axis must be 1 (x), 2 (y), or 3 (z)")
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
    # Initial state |00> as 4x1 column vector
    psi = np.array([[1], [0], [0], [0]], dtype=complex)
    
    # Apply R_x(π) to the second qubit to get |01> (up to global phase)
    gate_r2_x_pi = np.kron(np.eye(2), rotation_matrices(1, np.pi))
    psi = gate_r2_x_pi @ psi
    
    # Step 1: R_y(π/2) on qubit1, R_x(-π/2) on qubit2
    ry_pi2 = rotation_matrices(2, np.pi/2)
    rx_minus_pi2 = rotation_matrices(1, -np.pi/2)
    gate_step1 = np.kron(ry_pi2, rx_minus_pi2)
    psi = gate_step1 @ psi
    
    # Step 2: CNOT12 (control qubit1, target qubit2)
    gate_cnot12 = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0]], dtype=complex)
    psi = gate_cnot12 @ psi
    
    # Step 3: R_z(-2θ) on qubit2
    rz_minus2theta = rotation_matrices(3, -2 * theta)
    gate_step3 = np.kron(np.eye(2), rz_minus2theta)
    psi = gate_step3 @ psi
    
    # Step 4: CNOT12 again
    psi = gate_cnot12 @ psi
    
    # Step 5: R_y(-π/2) on qubit1, R_x(π/2) on qubit2
    ry_minus_pi2 = rotation_matrices(2, -np.pi/2)
    rx_pi2 = rotation_matrices(1, np.pi/2)
    gate_step5 = np.kron(ry_minus_pi2, rx_pi2)
    psi = gate_step5 @ psi
    
    return psi
