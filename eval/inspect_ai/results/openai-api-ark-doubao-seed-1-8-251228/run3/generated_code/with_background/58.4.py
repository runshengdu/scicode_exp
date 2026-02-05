import numpy as np
import scipy as sp
import scipy.integrate as si

def eos_press_from_rho(rho, eos_Gamma, eos_kappa):
    '''This function computes pressure for a polytropic equation of state given the density.
    Inputs:
    rho: the density, a float.
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    press: pressure corresponding to the given density, a float
    '''
    press = eos_kappa * (rho ** eos_Gamma)
    return press


def eos_rho_from_press(press, eos_Gamma, eos_kappa):
    '''This function computes density for a polytropic equation of state given the pressure.
    Inputs:
    press: pressure, a float
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    rho: density corresponding to the given pressure, a float
    '''
    rho = (press / eos_kappa) ** (1 / eos_Gamma)
    return rho


def eos_eps_from_press(press, eos_Gamma, eos_kappa):
    '''This function computes specific internal energy for a polytropic equation of state given the pressure.
    Inputs:
    press: the pressure, a float.
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    eps: the specific internal energy, a float.
    '''
    rho = eos_rho_from_press(press, eos_Gamma, eos_kappa)
    eps = press / (rho * (eos_Gamma - 1))
    return eps



def tov_RHS(data, r, eos_Gamma, eos_kappa):
    '''This function computes the integrand of the Tolman-Oppenheimer-Volkoff equation describing a neutron star consisting of a gas described by a polytropic equation of state.
    Inputs:
    data: the state vector, a 3-element tuple consiting of the current values for (`press`, `mass` and `phi`), all floats
    r: the radius at which to evaluate the right-hand-side
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    rhs: the integrand of the Tolman-Oppenheimer-Volkoff equation, a 3-element tuple of update terms for (`press`, `mass` and `phi`), all floats. 0 when outside of the star.
    '''
    press, mass, phi = data
    
    # Return zero derivatives outside the star (pressure <= 0)
    if press <= 0.0:
        return (0.0, 0.0, 0.0)
    
    # Handle center of the star (r=0) to avoid division by zero
    if r == 0.0:
        return (0.0, 0.0, 0.0)
    
    # Compute rest mass density from pressure using EOS
    rho = eos_rho_from_press(press, eos_Gamma, eos_kappa)
    # Compute specific internal energy from pressure using EOS
    eps = eos_eps_from_press(press, eos_Gamma, eos_kappa)
    # Compute total energy density
    mu = rho * (1 + eps)
    
    # Calculate common factor for pressure gradient and potential gradient
    numerator = mass + 4 * np.pi * r**3 * press
    denominator = r * (r - 2 * mass)
    factor = numerator / denominator
    
    # Compute TOV right-hand side derivatives
    dP_dr = - (mu + press) * factor
    dm_dr = 4 * np.pi * r**2 * mu
    dphi_dr = factor
    
    return (dP_dr, dm_dr, dphi_dr)
