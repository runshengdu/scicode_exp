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
    # Apply unitary transformation to the state
    psi_prime = U @ psi
    
    # Define the Z1 operator (Pauli Z on first qubit, identity on second)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    z1_operator = np.kron(pauli_z, np.eye(2, dtype=complex))
    
    # Calculate expectation value
    expectation = psi_prime.conj().T @ z1_operator @ psi_prime
    
    # Extract real scalar value (expectation of Hermitian operator is real)
    measured_result = np.real(expectation.item())
    
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
    # Get the ansatz wavefunction
    psi = create_ansatz(theta)
    
    # Term 0: Constant term g0 * <I>
    term0 = gl[0]
    
    # Term 1: g1 * <Z₁⊗I>
    U1 = np.eye(4, dtype=complex)
    exp_z1 = measureZ(U1, psi)
    term1 = gl[1] * exp_z1
    
    # Term 2: g2 * <I⊗Z₂> using SWAP gate
    swap = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]], dtype=complex)
    exp_z2 = measureZ(swap, psi)
    term2 = gl[2] * exp_z2
    
    # Term 3: g3 * <Z₁⊗Z₂> using CNOT₂₁ gate
    cnot21 = np.array([[1, 0, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0]], dtype=complex)
    exp_z1z2 = measureZ(cnot21, psi)
    term3 = gl[3] * exp_z1z2
    
    # Term 4: g4 * <Y₁⊗Y₂> using U = CNOT₂₁(HS†⊗HS†)
    # Define necessary single-qubit gates
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    Sdagger = S.conj().T
    HSdagger = H @ Sdagger
    # Construct the full unitary transformation
    hs_kron = np.kron(HSdagger, HSdagger)
    U4 = cnot21 @ hs_kron
    exp_y1y2 = measureZ(U4, psi)
    term4 = gl[4] * exp_y1y2
    
    # Term 5: g5 * <X₁⊗X₂> using U = CNOT₂₁(H⊗H)
    h_kron = np.kron(H, H)
    U5 = cnot21 @ h_kron
    exp_x1x2 = measureZ(U5, psi)
    term5 = gl[5] * exp_x1x2
    
    # Sum all contributions to get the total energy
    energy = term0 + term1 + term2 + term3 + term4 + term5
    
    return energy
