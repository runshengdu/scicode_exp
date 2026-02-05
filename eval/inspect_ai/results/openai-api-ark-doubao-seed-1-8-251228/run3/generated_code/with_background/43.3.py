import numpy as np
from scipy.integrate import solve_bvp


def f(z, y, Pssat, Ppsat, N, sigma_ap, sigma_ep, sigma_as, sigma_es, gamma_p, alpha_p, gamma_s, alpha_s):
    '''System of differential equations representing the rate equations of the fiber laser.
    Parameters:
    z : float
        Spatial variable along the fiber's length, representing the position.
    y : ndarray
        Array containing the power values of [forward pump, backward pump, forward signal, backward signal].
    Pssat : float
        Saturation power for the signal.
    Ppsat : float
        Saturation power for the pump.
    N : float
        Total ion population in the fiber.
    sigma_ap : float
        Absorption cross-section for the pump.
    sigma_ep : float
        Emission cross-section for the pump.
    sigma_as : float
        Absorption cross-section for the signal.
    sigma_es : float
        Emission cross-section for the signal.
    gamma_p : float
        Gain coefficient for the pump.
    alpha_p : float
        Loss coefficient for the pump.
    gamma_s : float
        Gain coefficient for the signal.
    alpha_s : float
        Loss coefficient for the signal.
    Returns: ndarray
    The rate of change of the power values for the pump and signal along the fiber:
        dydz[0]: Rate of change of forward pump power.
        dydz[1]: Rate of change of backward pump power.
        dydz[2]: Rate of change of forward signal power.
        dydz[3]: Rate of change of backward signal power.
    '''
    # Unpack the input power array
    pp_plus, pp_minus, ps_plus, ps_minus = y
    
    # Calculate total forward/backward combined powers
    pp_total = pp_plus + pp_minus
    ps_total = ps_plus + ps_minus
    
    # Compute numerator for N2/N ratio
    term1 = (sigma_ap / (sigma_ap + sigma_ep)) * (pp_total / Ppsat)
    term2 = (sigma_as / (sigma_as + sigma_es)) * (ps_total / Pssat)
    numerator = term1 + term2
    
    # Compute denominator for N2/N ratio
    denominator = (pp_total / Ppsat) + 1 + (ps_total / Pssat)
    
    # Calculate upper level population fraction and absolute population
    n2_over_n = numerator / denominator
    n2 = N * n2_over_n
    
    # Compute common terms for pump and signal rate equations
    pump_term = sigma_ap * N - (sigma_ap + sigma_ep) * n2
    signal_term = (sigma_es + sigma_as) * n2 - sigma_as * N
    
    # Calculate individual derivatives
    dpp_plus_dz = -gamma_p * pump_term * pp_plus - alpha_p * pp_plus
    dpp_minus_dz = gamma_p * pump_term * pp_minus + alpha_p * pp_minus
    dps_plus_dz = gamma_s * signal_term * ps_plus - alpha_s * ps_plus
    dps_minus_dz = -gamma_s * signal_term * ps_minus + alpha_s * ps_minus
    
    # Combine derivatives into output array
    dydz = np.array([dpp_plus_dz, dpp_minus_dz, dps_plus_dz, dps_minus_dz])
    
    return dydz


def bc(ya, yb, Ppl, Ppr, R1, R2):
    '''Define the boundary conditions for the fiber laser.
    Parameters:
    ya : ndarray
        Array of power values at the start of the fiber. Contains values corresponding to:
        ya[0] - Power of the forward pump at the fiber input.
        ya[1] - Power of the backward pump at the fiber input.
        ya[2] - Power of the forward signal at the fiber input.
        ya[3] - Power of the backward signal at the fiber input.
    yb : ndarray
        Array of power values at the end of the fiber. Contains values corresponding to:
        yb[0] - Power of the forward pump at the fiber output.
        yb[1] - Power of the backward pump at the fiber output.
        yb[2] - Power of the forward signal at the fiber output.
        yb[3] - Power of the backward signal at the fiber output.
    Ppl : float
        Input power for the left pump, affecting the starting boundary of the laser.
    Ppr : float
        Input power for the right pump, affecting the ending boundary of the laser.
    R1 : float
        Reflectivity of the input mirror, modifying the behavior of the light at the fiber's start.
    R2 : float
        Reflectivity of the output mirror, modifying the behavior of the light at the fiber's end.
    Returns:
    ndarray
        An array of four boundary conditions calculated as follows:
        bc[0]: boudary condition for rate of change of forward pump power.
        bc[1]: boudary condition for rate of change of backward pump power.
        bc[2]: boudary condition for rate of change of forward signal power.
        bc[3]: boudary condition for rate of change of backward signal power.
    '''
    # Calculate each boundary condition residual
    bc0 = ya[0] - Ppl
    bc1 = yb[1] - Ppr
    bc2 = ya[2] - R1 * ya[3]
    bc3 = yb[3] - R2 * yb[2]
    
    # Return the boundary conditions as an array
    return np.array([bc0, bc1, bc2, bc3])



def Pout_Nz_Calculation(lambda_s, lambda_p, tau, sigma_ap, sigma_ep, sigma_as, sigma_es, A_c, N, alpha_p, alpha_s, gamma_s, gamma_p, R1, R2, L, Ppl, Ppr):
    '''Calculate the output power and normalized population inversion along the length of the fiber.
    Parameters:
    lambda_s : float
        Wavelength of the signal in meters.
    lambda_p : float
        Wavelength of the pump in meters.
    tau : float
        Lifetime of the excited state in seconds.
    sigma_ap : float
        Absorption cross-section for the pump.
    sigma_ep : float
        Emission cross-section for the pump.
    sigma_as : float
        Absorption cross-section for the signal.
    sigma_es : float
        Emission cross-section for the signal.
    A_c : float
        Core area of the fiber in square meters.
    N : float
        Total ion population in the fiber.
    alpha_p : float
        Loss coefficient for the pump.
    alpha_s : float
        Loss coefficient for the signal.
    gamma_s : float
        Gain coefficient for the signal.
    gamma_p : float
        Gain coefficient for the pump.
    R1 : float
        Reflectivity of the input mirror.
    R2 : float
        Reflectivity of the output mirror.
    L : float
        Length of the fiber in meters.
    Ppl : float
        Input power for the left pump.
    Ppr : float
        Input power for the right pump.
    Returns:
    tuple (float, ndarray)
        Pout : float
            Output power of the signal at the end of the fiber, considering the output mirror losses.
        nz : ndarray
            Normalized population inversion distribution along the length of the fiber.
    '''
    # Nested function to calculate saturation power for pump or signal
    def calculate_saturation_power(lambda_val, sigma_total, tau_val, A_c_val):
        h = 6.626e-34  # Planck's constant in J*s
        c = 3e8         # Speed of light in m/s
        return (h * c * A_c_val) / (lambda_val * sigma_total * tau_val)
    
    # Calculate total cross-sections (absorption + emission) for pump and signal
    sigma_total_p = sigma_ap + sigma_ep
    sigma_total_s = sigma_as + sigma_es
    
    # Compute saturation powers using nested function
    Ppsat = calculate_saturation_power(lambda_p, sigma_total_p, tau, A_c)
    Pssat = calculate_saturation_power(lambda_s, sigma_total_s, tau, A_c)
    
    # Wrapper function for the rate equations (ODE system)
    def ode_fun(z, y):
        return f(z, y, Pssat, Ppsat, N, sigma_ap, sigma_ep, sigma_as, sigma_es, gamma_p, alpha_p, gamma_s, alpha_s)
    
    # Wrapper function for boundary conditions
    def bc_fun(ya, yb):
        return bc(ya, yb, Ppl, Ppr, R1, R2)
    
    # Create initial guess for BVP solution
    z_guess = np.linspace(0, L, 50)
    # Initial guesses for power components
    pp_plus_guess = Ppl * np.ones_like(z_guess)    # Forward pump starts at input power
    pp_minus_guess = Ppr * np.ones_like(z_guess)   # Backward pump starts at input power
    ps_plus_guess = 1e-6 * np.ones_like(z_guess)   # Small initial forward signal
    # Handle R1=0 case to avoid division by zero in initial guess
    ps_minus_guess = ps_plus_guess / R1 if R1 != 0 else 1e-6 * np.ones_like(z_guess)
    
    # Stack guesses into required shape (n_components, n_points)
    y_guess = np.vstack([pp_plus_guess, pp_minus_guess, ps_plus_guess, ps_minus_guess])
    
    # Solve the boundary value problem
    sol = solve_bvp(ode_fun, bc_fun, z_guess, y_guess)
    
    # Calculate output power (transmitted through output mirror)
    ps_plus_end = sol.sol(L)[2]
    Pout = (1 - R2) * ps_plus_end
    
    # Calculate normalized population inversion profile along fiber
    z_eval = np.linspace(0, L, 1000)
    y_eval = sol.sol(z_eval)
    pp_plus_eval, pp_minus_eval, ps_plus_eval, ps_minus_eval = y_eval
    
    # Compute combined powers at each position
    pp_total = pp_plus_eval + pp_minus_eval
    ps_total = ps_plus_eval + ps_minus_eval
    
    # Calculate upper level population fraction
    term1 = (sigma_ap / sigma_total_p) * (pp_total / Ppsat)
    term2 = (sigma_as / sigma_total_s) * (ps_total / Pssat)
    numerator = term1 + term2
    denominator = (pp_total / Ppsat) + 1 + (ps_total / Pssat)
    n2_over_n = numerator / denominator
    
    # Normalized population inversion: (N2 - N1)/N = 2*(N2/N) - 1
    nz = 2 * n2_over_n - 1
    
    return Pout, nz
