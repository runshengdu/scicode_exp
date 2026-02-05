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
    # Unpack the input array into individual power components
    pp_plus, pp_minus, ps_plus, ps_minus = y
    
    # Calculate total pump and signal powers
    pump_total = pp_plus + pp_minus
    signal_total = ps_plus + ps_minus
    
    # Compute numerator and denominator for N2(z)/N
    term_pump = (sigma_ap / (sigma_ap + sigma_ep)) * (pump_total / Ppsat)
    term_signal = (sigma_as / (sigma_as + sigma_es)) * (signal_total / Pssat)
    numerator = term_pump + term_signal
    
    denominator = (pump_total / Ppsat) + 1 + (signal_total / Pssat)
    
    # Calculate upper level population fraction and absolute population
    n2_over_n = numerator / denominator
    n2 = N * n2_over_n
    
    # Compute the common bracket terms for pump and signal equations
    bracket_pump = sigma_ap * N - (sigma_ap + sigma_ep) * n2
    bracket_signal = (sigma_es + sigma_as) * n2 - sigma_as * N
    
    # Calculate each derivative based on the rate equations
    d_pp_plus = -gamma_p * bracket_pump * pp_plus - alpha_p * pp_plus
    d_pp_minus = gamma_p * bracket_pump * pp_minus + alpha_p * pp_minus
    d_ps_plus = gamma_s * bracket_signal * ps_plus - alpha_s * ps_plus
    d_ps_minus = -gamma_s * bracket_signal * ps_minus + alpha_s * ps_minus
    
    # Package derivatives into the output array
    dydz = np.array([d_pp_plus, d_pp_minus, d_ps_plus, d_ps_minus])
    
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
    # Boundary condition for forward pump at fiber start
    bc0 = ya[0] - Ppl
    # Boundary condition for backward pump at fiber end
    bc1 = yb[1] - Ppr
    # Boundary condition for signal at input mirror
    bc2 = ya[2] - R1 * ya[3]
    # Boundary condition for signal at output mirror
    bc3 = yb[3] - R2 * yb[2]
    
    return np.array([bc0, bc1, bc2, bc3])
