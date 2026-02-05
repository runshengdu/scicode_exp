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
