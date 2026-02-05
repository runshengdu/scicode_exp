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
