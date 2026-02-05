import numpy as np


def f_V(q, d, bg_eps, l1, l2):
    '''Write down the form factor f(q;l1,l2)
    Input
    q, in-plane momentum, float in the unit of inverse angstrom
    d, layer spacing, float in the unit of angstrom
    bg_eps: LEG dielectric constant, float, dimensionless
    l1,l2: layer number where z = l*d, integer
    Output
    form_factor: form factor, float
    '''
    direct_term = np.exp(-q * d * abs(l1 - l2))
    alpha = (bg_eps - 1) / (bg_eps + 1)
    image_term = alpha * np.exp(-q * d * (l1 + l2))
    form_factor = direct_term + image_term
    return form_factor



def D_2DEG(q, omega, gamma, n_eff, e_F, k_F, v_F):
    '''Write down the exact form of density-density correlation function
    Input
    q, in-plane momentum, float in the unit of inverse angstrom
    omega, energy, real part, float in the unit of meV
    gamma, energy, imaginary part, float in the unit of meV
    n_eff, electron density, float in the unit of per square angstrom
    e_F, Fermi energy, float in the unit of meV
    k_F, Fermi momentum, float in the unit of inverse angstrom
    v_F, hbar * Fermi velocity, float in the unit of meV times angstrom
    Output
    D0: density-density correlation function, complex array in the unit of per square angstrom per meV
    '''
    # Handle q=0 case where the numerator of the integral is zero
    if np.isclose(q, 0.0):
        return 0.0 + 0.0j
    
    q_critical = 2 * k_F
    x = q / q_critical
    y = (omega + 1j * gamma) / (q * v_F)
    
    if q <= q_critical:
        sqrt_term = np.sqrt(1 - x**2)
        arctan_argument = sqrt_term / y
        arctan_value = np.arctan(arctan_argument)
        correction_term = (y / sqrt_term) * arctan_value
    else:
        sqrt_term = np.sqrt(x**2 - 1)
        log_numerator = y + sqrt_term
        log_denominator = y - sqrt_term
        log_argument = log_numerator / log_denominator
        log_value = np.log(log_argument)
        correction_term = (y / (2 * sqrt_term)) * log_value
    
    formula = 1 - correction_term
    D0 = (n_eff / e_F) * formula
    return D0


def D_cal(D0, q, d, bg_eps, N):
    '''Calculate the matrix form of density-density correlation function D(l1,l2)
    Input
    D0, density-density correlation function, complex array in the unit of per square angstrom per meV
    q, in-plane momentum, float in the unit of inverse angstrom
    d, layer spacing, float in the unit of angstrom
    bg_eps: LEG dielectric constant, float
    N: matrix dimension, integer
    Output
    D: NxN complex matrix, in the unit of per square angstrom per meV
    '''
    # Handle q=0 case where D0 is zero to avoid division by zero
    if np.isclose(q, 0.0):
        return np.zeros((N, N), dtype=np.complex128)
    
    # Construct the form factor matrix F
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    direct_term = np.exp(-q * d * np.abs(i - j))
    alpha = (bg_eps - 1) / (bg_eps + 1)
    image_term = alpha * np.exp(-q * d * (i + j))
    F = direct_term + image_term
    
    # Calculate 2D Coulomb interaction V_q using given vacuum dielectric constant
    epsilon0 = 55.26349406  # Vacuum dielectric constant in e² eV⁻¹ μm⁻¹
    e2_over_2epsilon0_eV_um = 1 / (2 * epsilon0)
    # Convert eV·μm to meV·Å: 1 eV=1000 meV, 1 μm=1e4 Å
    e2_over_2epsilon0_meV_ang = e2_over_2epsilon0_eV_um * 1000 * 1e4
    V_q = e2_over_2epsilon0_meV_ang / q  # V_q in meV·Å²
    
    # Construct interaction matrix V = V_q * F
    V_matrix = V_q * F
    
    # Construct non-interacting D0 matrix (diagonal matrix)
    D0_matrix = np.eye(N, dtype=np.complex128) * D0
    
    # Solve Dyson equation: D = (I - D0 @ V)⁻¹ @ D0
    M = np.eye(N, dtype=np.complex128) - D0_matrix @ V_matrix
    D = np.linalg.inv(M) @ D0_matrix
    
    return D



def D_l_analy(l1, l2, q, d, D0, bg_eps):
    '''Calculate the explicit form of density-density correlation function D(l1,l2) of semi-infinite LEG
    Input
    l1,l2: layer number where z = l*d, integer
    q, in-plane momentum, float in the unit of inverse angstrom
    d, layer spacing, float in the unit of angstrom
    D0, density-density correlation function, complex array in the unit of per square angstrom per meV
    bg_eps: LEG dielectric constant, float
    Output
    D_l: density-density correlation function, complex number in the unit of per square angstrom per meV
    '''
    K = q * d
    alpha = (bg_eps - 1) / (bg_eps + 1)
    
    # Calculate V_q (2D Coulomb interaction in meV·Å²)
    epsilon0 = 55.26349406  # e² eV⁻¹ μm⁻¹
    e2_over_2epsilon0_eV_um = 1 / (2 * epsilon0)
    # Convert eV·μm to meV·Å: 1 eV=1000 meV, 1 μm=1e4 Å
    e2_over_2epsilon0_meV_ang = e2_over_2epsilon0_eV_um * 1000 * 1e4
    V_q = e2_over_2epsilon0_meV_ang / q  # meV·Å²
    
    lambda_ = D0 * V_q
    
    # Define helper terms
    b = np.exp(-K)
    c = lambda_ * (1 - alpha * b**2)
    denom = 1 - 2 * lambda_ * b + lambda_**2 * b**2 - lambda_**2 * alpha**2 * b**2
    
    # Calculate direct and image contributions
    direct_term = np.exp(-K * np.abs(l1 - l2))
    image_term = alpha * np.exp(-K * (l1 + l2))
    
    # Calculate additional terms from the exact solution
    if l1 >= l2:
        term1 = (lambda_ * b)**(l1 - l2)
        term2 = alpha * (lambda_ * b * alpha)**(l1 + l2)
    else:
        term1 = (lambda_ * b)**(l2 - l1)
        term2 = alpha * (lambda_ * b * alpha)**(l1 + l2)
    
    # Exact form factor for the interacting case
    numerator = direct_term + lambda_ * (b**2 - alpha**2) * term1 + alpha * image_term + lambda_ * b * (1 - alpha**2) * term2
    f_interact = numerator / denom
    
    # Final correlation function
    D_l = D0 * f_interact
    
    return D_l


def omega_s_cal(q, gamma, n_eff, e_F, k_F, v_F, d, bg_eps):
    '''Calculate the surface plasmon of a semi-infinite LEG
    Input
    q, in-plane momentum, float in the unit of inverse angstrom
    gamma, energy, imaginary part, float in the unit of meV
    n_eff, electron density, float in the unit of per square angstrom
    e_F, Fermi energy, float in the unit of meV
    k_F, Fermi momentum, float in the unit of inverse angstrom
    v_F, hbar * Fermi velocity, float in the unit of meV times angstrom
    d, layer spacing, float in the unit of angstrom
    bg_eps: LEG dielectric constant, float
    Output
    omega_s: surface plasmon frequency, float in the unit of meV
    '''
    if np.isclose(q, 0.0):
        return 0.0  # No surface plasmon at q=0
    
    # Calculate 2D Coulomb interaction V_q in meV·Å²
    epsilon0 = 55.26349406  # Vacuum dielectric constant in e² eV⁻¹ μm⁻¹
    e2_over_2epsilon0_eV_um = 1 / (2 * epsilon0)
    # Convert units from eV·μm to meV·Å
    e2_over_2epsilon0_meV_ang = e2_over_2epsilon0_eV_um * 1000 * 1e4
    V_q = e2_over_2epsilon0_meV_ang / q
    
    alpha = (bg_eps - 1) / (bg_eps + 1)
    b = np.exp(-q * d)
    C = V_q * b * (1 + alpha)
    
    # Target function to find root of: 1 - D0(omega) * C = 0
    def target(omega):
        d0 = D_2DEG(q, omega, gamma, n_eff, e_F, k_F, v_F)
        return 1 - d0 * C
    
    # Numerical derivative of target function using finite difference
    def derivative(omega, h=1e-3):
        return (target(omega + h) - target(omega - h)) / (2 * h)
    
    # Estimate initial guess range for omega
    q_critical = 2 * k_F
    if q <= q_critical:
        # Minimum single-particle excitation energy for q <= 2k_F
        omega_min = e_F * q * (2 * k_F - q) / (k_F ** 2)
    else:
        # Minimum single-particle excitation energy for q > 2k_F
        omega_min = (q ** 2 * e_F) / (k_F ** 2) - q * v_F
    # Upper bound set to 5 times maximum single-particle excitation energy
    omega_max = 5 * e_F * (1 + q / k_F) ** 2
    omega_guess = (omega_min + omega_max) / 2
    
    # Newton-Raphson iteration to find root
    tolerance = 1e-6
    max_iter = 100
    for _ in range(max_iter):
        f_val = target(omega_guess)
        if np.abs(f_val) < tolerance:
            break
        df_val = derivative(omega_guess)
        if np.abs(df_val) < float(1e-12):
            break  # Avoid division by zero if derivative is too small
        omega_guess -= f_val / df_val
    
    # Return the real part as the surface plasmon frequency
    return np.real(omega_guess)


def I_Raman(q, d, omega, gamma, n_eff, e_F, k_F, v_F, bg_eps, delta_E, kd):
    '''Calculate the Raman intensity
    Input
    q, in-plane momentum, float in the unit of inverse angstrom
    d, layer spacing, float in the unit of angstrom
    omega, energy, real part, float in the unit of meV
    gamma, energy, imaginary part, float in the unit of meV
    n_eff, electron density, float in the unit of per square angstrom
    e_F, Fermi energy, float in the unit of meV
    k_F, Fermi momentum, float in the unit of inverse angstrom
    v_F, hbar * Fermi velocity, float in the unit of meV times angstrom
    bg_eps: LEG dielectric constant, float
    delta_E: penetration depth, float in the unit of angstrom
    kd: wave number times layer spacing, float dimensionless
    Output
    I_omega: Raman intensity, float
    '''
    if np.isclose(q, 0.0):
        return 0.0
    
    # Compute non-interacting density-density correlation function
    D0 = D_2DEG(q, omega, gamma, n_eff, e_F, k_F, v_F)
    
    # Calculate 2D Coulomb interaction V_q in meV·Å²
    epsilon0 = 55.26349406  # Vacuum dielectric constant in e² eV⁻¹ μm⁻¹
    e2_over_2epsilon0_eV_um = 1 / (2 * epsilon0)
    # Convert eV·μm to meV·Å: 1 eV=1000 meV, 1 μm=1e4 Å
    e2_over_2epsilon0_meV_ang = e2_over_2epsilon0_eV_um * 1000 * 1e4
    V_q = e2_over_2epsilon0_meV_ang / q  # meV·Å²
    
    # Define key parameters
    K = q * d
    alpha = (bg_eps - 1) / (bg_eps + 1)
    b = np.exp(-K)
    beta = d / delta_E
    lambda_ = D0 * V_q
    cos_2kd = np.cos(2 * kd)
    
    # Denominator from analytical solution of D(l1,l2)
    denom = 1 - 2 * lambda_ * b + (lambda_ * b) ** 2 * (1 - alpha ** 2)
    
    # Calculate S3 term
    b1 = np.exp(-(K + beta))
    denom_S3 = 1 - 2 * b1 * cos_2kd + b1 ** 2
    S3 = (alpha ** 2) / denom_S3
    
    # Calculate S4 term
    s4_num = alpha * lambda_ * b * (1 - alpha ** 2)
    s_S4 = lambda_ * b * np.exp(-beta)
    denom_S4 = 1 - 2 * s_S4 * cos_2kd + s_S4 ** 2
    S4 = s4_num / denom_S4
    
    # Calculate S1 term
    denominator_S1_part = -np.expm1(-2 * beta)  # Equivalent to 1 - np.exp(-2*beta)
    numerator_S1 = 1 - b1 ** 2
    denom_S1 = denominator_S1_part * (1 - 2 * b1 * cos_2kd + b1 ** 2)
    S1 = numerator_S1 / denom_S1
    
    # Calculate S2 term
    s2_coeff = lambda_ * (b ** 2 - alpha ** 2)
    s_S2 = lambda_ * b * np.exp(-beta)
    numerator_S1_prime = 1 - s_S2 ** 2
    denom_S1_prime = denominator_S1_part * (1 - 2 * s_S2 * cos_2kd + s_S2 ** 2)
    S1_prime = numerator_S1_prime / denom_S1_prime
    S2 = s2_coeff * S1_prime
    
    # Total sum S
    S_total = (D0 / denom) * (S1 + S2 + S3 + S4)
    
    # Extract Raman intensity from imaginary part
    I_omega = -np.imag(S_total)
    
    return I_omega


def I_Raman_eval(q, d, omega, gamma, n_eff, e_F, k_F, v_F, bg_eps, delta_E, kd):
    '''Calculate the Raman intensity
    Input
    q, in-plane momentum, float in the unit of inverse angstrom
    d, layer spacing, float in the unit of angstrom
    omega, energy, real part, float in the unit of meV
    gamma, energy, imaginary part, float in the unit of meV
    n_eff, electron density, float in the unit of per square angstrom
    e_F, Fermi energy, float in the unit of meV
    k_F, Fermi momentum, float in the unit of inverse angstrom
    v_F, hbar * Fermi velocity, float in the unit of meV times angstrom
    bg_eps: LEG dielectric constant, float
    delta_E: penetration depth, float in the unit of angstrom
    kd: wave number times layer spacing, float dimensionless
    Output
    I_omega: Raman intensity, float
    '''
    if np.isclose(q, 0.0):
        return 0.0
    
    # Compute non-interacting correlation function D0
    D0 = D_2DEG(q, omega, gamma, n_eff, e_F, k_F, v_F)
    
    # Calculate 2D Coulomb interaction V_q in meV·Å²
    epsilon0 = 55.26349406  # Vacuum dielectric constant in e² eV⁻¹ μm⁻¹
    e2_over_2epsilon0_eV_um = 1 / (2 * epsilon0)
    # Convert eV·μm to meV·Å: 1 eV=1000 meV, 1 μm=1e4 Å
    e2_over_2epsilon0_meV_ang = e2_over_2epsilon0_eV_um * 1000 * 1e4
    V_q = e2_over_2epsilon0_meV_ang / q  # meV·Å²
    
    alpha = (bg_eps - 1) / (bg_eps + 1)
    qd = q * d
    
    # Hyperbolic and exponential terms
    sinh_qd = np.sinh(qd)
    cosh_qd = np.cosh(qd)
    exp_qd = np.exp(qd)
    exp_mqd = np.exp(-qd)
    exp_2qd = np.exp(2 * qd)
    
    # Compute b from formula
    b_formula = cosh_qd - D0 * V_q * sinh_qd
    b_squared_minus_1 = b_formula ** 2 - 1
    
    # Compute sqrt(b²-1) with positive imaginary part
    sqrt_b2_1 = np.sqrt(b_squared_minus_1)
    if np.imag(sqrt_b2_1) < 0:
        sqrt_b2_1 = -sqrt_b2_1
    
    u = b_formula + sqrt_b2_1
    
    # Compute G
    term_G1 = 1 / sqrt_b2_1
    term_G2 = 1 / sinh_qd
    G = 0.5 * (term_G1 - term_G2) / sinh_qd
    
    # Compute H
    term_H1 = (1 / u) / sqrt_b2_1
    term_H2 = exp_mqd / sinh_qd
    H = 0.5 * (term_H1 - term_H2) / sinh_qd
    
    sinh_qd_sq = sinh_qd ** 2
    
    # Compute A, B, C
    A = G * sinh_qd_sq + 1 + 0.5 * alpha * exp_2qd
    B = H * sinh_qd_sq + cosh_qd + 0.5 * alpha * exp_qd
    C = G * sinh_qd_sq + 1 + 0.5 * alpha
    
    # Compute Q
    term_Q1 = 1 - (1 - b_formula * cosh_qd) / (sqrt_b2_1 * sinh_qd)
    term_Q2 = alpha * exp_qd * (cosh_qd - b_formula) / (sqrt_b2_1 * sinh_qd)
    Q = 0.5 * term_Q1 - 0.5 * term_Q2
    
    # Compute E
    exp_2d_delta = np.exp(2 * d / delta_E)
    exp_d_delta = np.exp(d / delta_E)
    cos_2kd = np.cos(2 * kd)
    E = (u ** 2) * exp_2d_delta + 1 - 2 * u * exp_d_delta * cos_2kd
    
    # Compute term1_total
    inv_1_minus_exp_m2d_delta = 1 / (1 - np.exp(-2 * d / delta_E))
    term1_num = D0 * V_q * sinh_qd * ((u ** 2) * exp_2d_delta - 1)
    term1_den = sqrt_b2_1 * E
    term1_bracket = 1 + term1_num / term1_den
    term1_total = inv_1_minus_exp_m2d_delta * term1_bracket
    
    # Compute term2_total
    numerator_term2 = D0 * V_q * exp_2d_delta * ((u ** 2) * A - 2 * u * B + C)
    denominator_term2 = 2 * Q * b_squared_minus_1 * E
    term2_total = numerator_term2 / denominator_term2
    
    # Total expression inside imaginary part
    total_expression = D0 * (term1_total + term2_total)
    
    # Calculate Raman intensity
    I_omega = -np.imag(total_expression)
    
    return I_omega



def I_Raman_num(q, d, omega, gamma, n_eff, e_F, k_F, v_F, bg_eps, delta_E, kd, N):
    '''Calculate the Raman intensity
    Input
    q, in-plane momentum, float in the unit of inverse angstrom
    d, layer spacing, float in the unit of angstrom
    omega, energy, real part, float in the unit of meV
    gamma, energy, imaginary part, float in the unit of meV
    n_eff, electron density, float in the unit of per square angstrom
    e_F, Fermi energy, float in the unit of meV
    k_F, Fermi momentum, float in the unit of inverse angstrom
    v_F, hbar * Fermi velocity, float in the unit of meV times angstrom
    bg_eps: LEG dielectric constant, float
    delta_E: penetration depth, float in the unit of angstrom
    kd: wave number times layer spacing, float dimensionless
    N: matrix dimension, integer
    Output
    I_omega_num: Raman intensity, float
    '''
    if np.isclose(q, 0.0):
        return 0.0
    
    # Compute non-interacting density-density correlation function
    D0 = D_2DEG(q, omega, gamma, n_eff, e_F, k_F, v_F)
    
    # Compute interacting correlation matrix using RPA
    D_matrix = D_cal(D0, q, d, bg_eps, N)
    
    # Create meshgrid for layer indices (0-based)
    l1, l2 = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    
    # Calculate exponential attenuation and phase factors
    exp_attenuation = np.exp(-d * (l1 + l2) / delta_E)
    exp_phase = np.exp(-2j * kd * (l1 - l2))
    exp_factor = exp_attenuation * exp_phase
    
    # Compute total sum of correlation matrix elements weighted by exponential factors
    total_sum = np.sum(D_matrix * exp_factor)
    
    # Extract Raman intensity from imaginary part of the total sum
    I_omega_num = -np.imag(total_sum)
    
    return I_omega_num
