import numpy as np
import scipy

def Rlnm(l, n, m, k, z, N_t):
    def compute_a(n_val, m_val):
        abs_m = abs(m_val)
        if n_val < abs_m:
            return 0.0
        numerator = (n_val + 1 + abs_m) * (n_val + 1 - abs_m)
        denominator = (2 * n_val + 1) * (2 * n_val + 3)
        return np.sqrt(numerator / denominator)
    
    def compute_b(n_val, m_val):
        abs_m = abs(m_val)
        if n_val < abs_m:
            return 0.0
        term1 = (n_val - m_val - 1)
        term2 = (n_val - m_val)
        numerator = term1 * term2
        denominator = (2 * n_val - 1) * (2 * n_val + 1)
        sqrt_val = np.sqrt(numerator / denominator)
        return sqrt_val if m_val >= 0 else -sqrt_val
    
    # Edge case handling
    abs_m = abs(m)
    if abs_m > l or abs_m > n:
        return 0.0 + 0.0j
    if l >= N_t or n >= N_t or N_t <= 0:
        return 0.0 + 0.0j
    
    r0 = z
    x = k * r0
    
    # Compute diagonal terms (R|R)_{l s}^s for s from 0 to abs_m
    diag_T = []
    # Initialize s=0
    s0 = np.zeros(N_t, dtype=np.complex128)
    for l_val in range(N_t):
        j_l = scipy.special.spherical_jn(l_val, x)
        s0[l_val] = ((-1) ** l_val) * np.sqrt(2 * l_val + 1) * j_l
    diag_T.append(s0)
    
    # Compute diag_T for s from 1 to abs_m
    for s in range(1, abs_m + 1):
        current_T = np.zeros(N_t, dtype=np.complex128)
        for l_val in range(s, N_t):
            b_m1_neg_m1 = compute_b(s, -s)
            if b_m1_neg_m1 == 0:
                continue  # Avoid division by zero (should not happen for s >=1)
            
            # Get (R|R)_{l-1, s-1}^{s-1}
            T_prev = diag_T[s-1][l_val-1] if (l_val - 1) >= 0 else 0.0
            b_l_neg_m1 = compute_b(l_val, -s)
            
            # Get (R|R)_{l+1, s-1}^{s-1}
            T_next = diag_T[s-1][l_val+1] if (l_val + 1) < N_t else 0.0
            b_l1_m = compute_b(l_val + 1, s - 1)
            
            # Solve for current_T[l_val]
            numerator = b_l_neg_m1 * T_prev - b_l1_m * T_next
            current_T[l_val] = numerator / b_m1_neg_m1
        diag_T.append(current_T)
    
    # Build T_plus matrix for (R|R)_{l n}^{abs_m}
    T_plus = np.zeros((N_t, N_t), dtype=np.complex128)
    # Fill column n=abs_m
    for l_val in range(abs_m, N_t):
        T_plus[l_val][abs_m] = diag_T[abs_m][l_val]
    
    # Fill remaining columns using recursion
    for n_val in range(abs_m, N_t - 1):
        a_n = compute_a(n_val, abs_m)
        if a_n == 0:
            continue  # Should not happen for n_val >= abs_m
        a_n_minus_1 = compute_a(n_val - 1, abs_m) if (n_val - 1) >= 0 else 0.0
        
        for l_val in range(abs_m, N_t):
            a_l = compute_a(l_val, abs_m)
            a_l_minus_1 = compute_a(l_val - 1, abs_m) if (l_val - 1) >= abs_m else 0.0
            
            # Get T[l_val][n_val-1]
            T_l_n_minus_1 = T_plus[l_val][n_val - 1] if (n_val - 1) >= abs_m else 0.0
            # Get T[l_val+1][n_val]
            T_l_plus_1_n = T_plus[l_val + 1][n_val] if (l_val + 1) < N_t else 0.0
            # Get T[l_val-1][n_val]
            T_l_minus_1_n = T_plus[l_val - 1][n_val] if (l_val - 1) >= abs_m else 0.0
            
            numerator = a_n_minus_1 * T_l_n_minus_1 - a_l * T_l_plus_1_n + a_l_minus_1 * T_l_minus_1_n
            T_plus[l_val][n_val + 1] = numerator / a_n
    
    # Apply symmetry for negative m
    if m >= 0:
        result = T_plus[l][n]
    else:
        result = ((-1) ** abs_m) * np.conj(T_plus[l][n])
    
    return result


def Tnvm(n, v, m, Q):
    '''Function to calculate the rotation coefficient Tnvm.
    Input
    n : int
        The principal quantum number for both expansion and reexpansion.
    m : int
        The magnetic quantum number for the reexpansion.
    v : int
        The magnetic quantum number for the expansion.
    Q : matrix of shape(3, 3)
        The rotation matrix.
    Output
    T : complex
        The rotation coefficient Tnvm.
    '''
    def compute_a(n_val, m_val):
        abs_m = abs(m_val)
        if n_val < abs_m:
            return 0.0
        numerator = (n_val + 1 + abs_m) * (n_val + 1 - abs_m)
        denominator = (2 * n_val + 1) * (2 * n_val + 3)
        return np.sqrt(numerator / denominator)
    
    def compute_b(n_val, m_val):
        abs_m = abs(m_val)
        if n_val < abs_m:
            return 0.0
        term1 = n_val - m_val - 1
        term2 = n_val - m_val
        numerator = term1 * term2
        denominator = (2 * n_val - 1) * (2 * n_val + 1)
        sqrt_val = np.sqrt(numerator / denominator)
        return sqrt_val if m_val >= 0 else -sqrt_val
    
    # Edge case handling for invalid quantum numbers
    if abs(v) > n or abs(m) > n:
        return 0.0 + 0.0j
    
    # Handle negative m using symmetry relation
    if m < 0:
        m_pos = -m
        v_neg = -v
        t_pos = Tnvm(n, v_neg, m_pos, Q)
        phase_factor = (-1) ** (v + m)
        return phase_factor * np.conj(t_pos)
    
    # Base case: m = 0
    if m == 0:
        # Extract components of original z-axis in rotated frame
        x_prime = Q[0][2]
        y_prime = Q[1][2]
        z_prime = Q[2][2]
        
        # Clamp z_prime to avoid numerical issues with arccos
        z_prime_clamped = np.clip(z_prime, -1.0, 1.0)
        theta_prime = np.arccos(z_prime_clamped)
        phi_prime = np.arctan2(y_prime, x_prime)
        
        # Compute spherical harmonic Y_n^{-v}(theta', phi')
        y_n_minus_v = scipy.special.sph_harm(-v, n, phi_prime, theta_prime)
        # Calculate the base case coefficient
        sqrt_factor = np.sqrt(4 * np.pi / (2 * n + 1))
        return sqrt_factor * y_n_minus_v
    else:
        # Recursive case for m >= 1
        # Compute recursion coefficients from rotation matrix
        term1_coeff = Q[0][0] + Q[1][1] + 1j * (Q[0][1] - Q[1][0])
        term2_coeff = Q[0][0] - Q[1][1] + 1j * (Q[0][1] + Q[1][0])
        term3_coeff = -2 * (Q[2][0] + 1j * Q[2][1])
        
        # Get required b and a coefficients
        b_n_neg_m = compute_b(n, -m)
        b_n_neg_v = compute_b(n, -v)
        b_n_v = compute_b(n, v)
        a_nm1_v = compute_a(n-1, v)
        
        # Recursively compute required lower-order coefficients
        t1 = Tnvm(n-1, v-1, m-1, Q)
        t2 = Tnvm(n-1, v+1, m-1, Q)
        t3 = Tnvm(n-1, v, m-1, Q)
        
        # Calculate numerator terms
        term1 = term1_coeff * b_n_neg_v * t1
        term2 = term2_coeff * b_n_v * t2
        term3 = term3_coeff * a_nm1_v * t3
        
        # Compute final value using recursion formula
        numerator = term1 + term2 + term3
        denominator = 2 * b_n_neg_m
        return numerator / denominator



def compute_BRnm(r0, B, n, m, wl, N_t):
    '''Function to calculate the reexpansion coefficient BR_nm.
    Input
    r_0 : array
        Translation vector.
    B : matrix of shape(N_t + 1, 2 * N_t + 1)
        Expansion coefficients of the elementary regular solutions.
    n : int
        The principal quantum number for the reexpansion.
    m : int
        The magnetic quantum number for the reexpansion.
    wl : float
        Wavelength of the optical beam.
    N_t : int
        Truncated space size.
    Output
    BR_nm : complex
        Reexpansion coefficient BR_nm of the elementary regular solutions.
    '''

    # Edge case: magnetic quantum number out of valid range
    if abs(m) > n:
        return 0.0 + 0.0j

    # Edge case: principal quantum number exceeds truncation size
    if n > N_t:
        return 0.0 + 0.0j

    # Calculate magnitude of translation vector
    r0_mag = np.linalg.norm(r0)

    # Edge case: no translation, return original coefficient
    if r0_mag < 1e-12:
        s_idx = m + N_t
        if 0 <= s_idx < 2 * N_t + 1:
            return B[n][s_idx].astype(np.complex128)
        else:
            return 0.0 + 0.0j

    # Compute wave number
    k = 2 * np.pi / wl

    # Unit vector along translation direction
    u_r0 = r0 / r0_mag
    u_z = np.array([0.0, 0.0, 1.0])

    # Construct rotation matrix P that rotates z-axis to translation direction
    if np.allclose(u_r0, u_z):
        P = np.eye(3, dtype=np.float64)
    elif np.allclose(u_r0, -u_z):
        P = np.array([[1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0],
                      [0.0, 0.0, -1.0]], dtype=np.float64)
    else:
        # Rodrigues' rotation formula components
        axis = np.cross(u_z, u_r0)
        axis_norm = np.linalg.norm(axis)
        axis_unit = axis / axis_norm
        cos_theta = np.clip(np.dot(u_z, u_r0), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        sin_theta = np.sin(theta)

        # Skew-symmetric matrix for rotation axis
        ax, ay, az = axis_unit
        skew = np.array([
            [0.0, -az, ay],
            [az, 0.0, -ax],
            [-ay, ax, 0.0]
        ], dtype=np.float64)

        # Build rotation matrix
        P = (np.eye(3, dtype=np.float64) * cos_theta +
             skew * sin_theta +
             np.outer(axis_unit, axis_unit) * (1.0 - cos_theta))

    # Q is inverse of P (transpose of orthogonal matrix)
    Q = P.T
    # Q inverse is transpose of Q (equal to original P)
    Q_inv = Q.T

    BRnm = 0.0 + 0.0j

    # Iterate over all possible l (principal quantum number of original expansion)
    for l in range(N_t + 1):
        # Iterate over valid magnetic quantum numbers for current l
        for s in range(-l, l + 1):
            s_idx = s + N_t
            # Skip out-of-bounds column indices (should not occur given l <= N_t)
            if not (0 <= s_idx < 2 * N_t + 1):
                continue

            B_ls = B[l][s_idx]
            # Skip zero coefficients for computational efficiency
            if np.isclose(B_ls, 0.0):
                continue

            # Iterate over all valid intermediate quantum numbers ν
            max_abs_nu = min(l, n)
            for nu in range(-max_abs_nu, max_abs_nu + 1):
                # Calculate rotation coefficients
                T_l_snu = Tnvm(l, s, nu, Q)
                T_n_num = Tnvm(n, nu, m, Q_inv)

                # Calculate translation coefficient
                RR_l_n_nu = Rlnm(l, n, nu, k, r0_mag, N_t)

                # Accumulate contribution to BRnm
                BRnm += B_ls * T_l_snu * T_n_num * RR_l_n_nu

    return BRnm
