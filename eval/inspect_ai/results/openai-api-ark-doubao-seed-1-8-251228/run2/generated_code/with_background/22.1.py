import numpy as np
import scipy





def Rlnm(l, n, m, k, z, N_t):
    '''Function to calculate the translation coefficient (R|R)lmn.
    Input
    l : int
        The principal quantum number for the expansion.
    n : int
        The principal quantum number for the reexpansion.
    m : int
        The magnetic quantum number for the reexpansion.
    k : float
        Wavevector of the optical beam.
    z : float
        The translation distance along z direction
    N_t : int
        Truncated space size.
    Output
    (R|R)lmn : complex
        The translation coefficient (R|R)lmn.
    '''
    def compute_a(n_val, m_val):
        m_abs = abs(m_val)
        if n_val < m_abs:
            return 0.0
        numerator = (n_val + 1 + m_abs) * (n_val + 1 - m_abs)
        denominator = (2 * n_val + 1) * (2 * n_val + 3)
        return np.sqrt(numerator / denominator)
    
    def compute_b(n_val, m_val):
        m_abs = abs(m_val)
        if n_val < m_abs:
            return 0.0
        numerator = (n_val - m_val - 1) * (n_val - m_val)
        denominator = (2 * n_val - 1) * (2 * n_val + 1)
        if numerator < 0 or denominator == 0:
            return 0.0
        sqrt_val = np.sqrt(numerator / denominator)
        return sqrt_val if m_val >= 0 else -sqrt_val
    
    m_abs = abs(m)
    # Boundary conditions: return 0 if indices are out of valid range
    if l < m_abs or n < m_abs or l > N_t or n > N_t:
        return complex(0.0)
    
    kr0 = k * z
    rr_list = []
    
    # Initialize for m' = 0 (magnetic quantum number 0)
    rr_m0 = np.zeros((N_t + 1, N_t + 1), dtype=np.float64)
    for l_val in range(N_t + 1):
        sign = (-1) ** l_val
        sqrt_term = np.sqrt(2 * l_val + 1)
        j_l = scipy.special.spherical_jn(l_val, kr0)
        rr_m0[l_val][0] = sign * sqrt_term * j_l
    
    # Compute all n values for m' = 0 using recursion
    for n_val in range(N_t):
        for l_val in range(N_t + 1):
            # Calculate RHS of the recursion relation
            a_l = compute_a(l_val, 0)
            term1 = a_l * (rr_m0[l_val + 1][n_val] if (l_val + 1 <= N_t) else 0.0)
            a_l_minus_1 = compute_a(l_val - 1, 0)
            term2 = a_l_minus_1 * (rr_m0[l_val - 1][n_val] if (l_val - 1 >= 0) else 0.0)
            rhs = term1 - term2
            
            # Calculate left-hand side components
            a_n_minus_1 = compute_a(n_val - 1, 0)
            term3 = a_n_minus_1 * (rr_m0[l_val][n_val - 1] if (n_val - 1 >= 0) else 0.0)
            a_n = compute_a(n_val, 0)
            
            if a_n != 0.0:
                rr_m0[l_val][n_val + 1] = (term3 - rhs) / a_n
    rr_list.append(rr_m0)
    
    # Compute for m' from 1 to m_abs
    for m_prime in range(1, m_abs + 1):
        rr_prev = rr_list[m_prime - 1]
        rr_current = np.zeros((N_t + 1, N_t + 1), dtype=np.float64)
        denom = compute_b(m_prime, -m_prime)
        
        # Compute (R|R)_{l, m_prime}^{m_prime} using the first recursion relation
        for l_val in range(m_prime, N_t + 1):
            b_l = compute_b(l_val, -m_prime)
            term1 = b_l * (rr_prev[l_val - 1][m_prime - 1] if (l_val - 1 >= 0) else 0.0)
            
            b_l_plus_1 = compute_b(l_val + 1, m_prime - 1)
            term2 = b_l_plus_1 * (rr_prev[l_val + 1][m_prime - 1] if (l_val + 1 <= N_t) else 0.0)
            
            numerator = term1 - term2
            rr_current[l_val][m_prime] = numerator / denom
        
        # Compute all n values for current m_prime using the second recursion relation
        for n_val in range(m_prime, N_t):
            for l_val in range(N_t + 1):
                if l_val < m_prime:
                    rr_current[l_val][n_val + 1] = 0.0
                    continue
                
                # Calculate RHS of the recursion relation
                a_l = compute_a(l_val, m_prime)
                term1 = a_l * (rr_current[l_val + 1][n_val] if (l_val + 1 <= N_t) else 0.0)
                a_l_minus_1 = compute_a(l_val - 1, m_prime)
                term2 = a_l_minus_1 * (rr_current[l_val - 1][n_val] if (l_val - 1 >= 0) else 0.0)
                rhs = term1 - term2
                
                # Calculate left-hand side components
                a_n_minus_1 = compute_a(n_val - 1, m_prime)
                term3 = a_n_minus_1 * (rr_current[l_val][n_val - 1] if (n_val - 1 >= 0) else 0.0)
                a_n = compute_a(n_val, m_prime)
                
                if a_n != 0.0:
                    rr_current[l_val][n_val + 1] = (term3 - rhs) / a_n
        
        rr_list.append(rr_current)
    
    # Get the result and adjust for negative m using symmetry assumption
    result = rr_list[m_abs][l][n]
    if m < 0:
        result = (-1) ** m * result
    
    return complex(result)
