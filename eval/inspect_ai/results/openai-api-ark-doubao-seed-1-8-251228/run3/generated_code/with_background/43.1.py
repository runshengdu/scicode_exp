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
