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
    
    # Define the Z⊗I operator for the first qubit
    Z1 = np.diag([1, 1, -1, -1]).astype(np.complex128)
    
    # Calculate the expectation value ⟨ψ'|Z1|ψ'⟩
    expectation = np.conj(psi_prime.T) @ Z1 @ psi_prime
    
    # Extract the real part (expectation of Hermitian operator is real)
    measured_result = float(np.real(expectation))
    
    return measured_result


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
    g0, g1, g2, g3, g4, g5 = gl
    psi = create_ansatz(theta)
    
    # Term 1: Z1⊗I, U is identity matrix
    U1 = np.eye(4, dtype=np.complex128)
    exp_z1 = measureZ(U1, psi)
    
    # Term 2: I⊗Z2, U is SWAP matrix
    U_swap = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=np.complex128)
    exp_z2 = measureZ(U_swap, psi)
    
    # Term 3: Z1⊗Z2, U is CNOT21 matrix
    U_cnot21 = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ], dtype=np.complex128)
    exp_z1z2 = measureZ(U_cnot21, psi)
    
    # Term 4: Y1⊗Y2, U = CNOT21 (HS† ⊗ HS†)
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    S_dagger = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
    HS_dagger = H @ S_dagger
    HS_dagger_kron = np.kron(HS_dagger, HS_dagger)
    U4 = U_cnot21 @ HS_dagger_kron
    exp_y1y2 = measureZ(U4, psi)
    
    # Term 5: X1⊗X2, U = CNOT21 (H⊗H)
    H_kron = np.kron(H, H)
    U5 = U_cnot21 @ H_kron
    exp_x1x2 = measureZ(U5, psi)
    
    # Calculate total energy
    energy = g0 + g1 * exp_z1 + g2 * exp_z2 + g3 * exp_z1z2 + g4 * exp_y1y2 + g5 * exp_x1x2
    
    return energy



def perform_vqe(gl):
    '''Perform vqe optimization
    Input:
    gl = [g0, g1, g2, g3, g4, g5] : array in size 6
        Hamiltonian coefficients.
    Output:
    energy : float
        VQE energy.
    '''
    # Define the cost function to minimize
    def cost(theta):
        return projective_expected(theta, gl)
    
    # Initial guess for the variational parameter theta
    initial_theta = 0.0
    
    # Perform minimization using scipy's optimize.minimize
    optimization_result = minimize(cost, x0=initial_theta, method='BFGS')
    
    # Extract the minimal energy value from the optimization result
    energy = optimization_result.fun
    
    return energy
