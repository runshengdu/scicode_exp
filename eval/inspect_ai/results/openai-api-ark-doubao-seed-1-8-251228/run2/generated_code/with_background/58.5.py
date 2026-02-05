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
    rho: density corresponding to the given pressure, a float.
    '''
    rho = (press / eos_kappa) ** (1.0 / eos_Gamma)
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
    rho = (press / eos_kappa) ** (1.0 / eos_Gamma)
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
    
    # Return zero derivatives outside the star (pressure is zero or negative)
    if press <= 0.0:
        return (0.0, 0.0, 0.0)
    
    # Compute rest mass density from pressure using polytropic EOS
    rho = eos_rho_from_press(press, eos_Gamma, eos_kappa)
    # Compute specific internal energy from pressure using Gamma-law EOS
    eps = eos_eps_from_press(press, eos_Gamma, eos_kappa)
    # Compute total energy density
    mu = rho * (1 + eps)
    
    # Calculate common factor terms for derivatives
    numerator_factor = mass + 4 * np.pi * (r ** 3) * press
    denominator_factor = r * (r - 2 * mass)
    
    # Avoid division by zero (physically irrelevant for neutron stars)
    if denominator_factor == 0.0:
        return (0.0, 0.0, 0.0)
    
    common_factor = numerator_factor / denominator_factor
    
    # Compute each component of the RHS vector
    dPdr = - (mu + press) * common_factor
    dmdr = 4 * np.pi * (r ** 2) * mu
    dphidr = common_factor
    
    return (dPdr, dmdr, dphidr)



def tov(rhoc, eos_Gamma, eos_kappa, npoints, rmax):
    '''This function computes gravitational time dilation at the center of the neutron star described by a polytropic equation of state as well as the star's mass.
    Inputs
    rhoc: float, the density at the center of the star, in units where G=c=Msun=1.
    Gamma: float, adiabatic exponent of the equation of state
    kappa: float, coefficient of the equation of state
    npoints: int, number of intergration points to use
    rmax: float, maximum radius to which to intgrate solution to, must include the whole star
    Outputs
    mass: float, gravitational mass of neutron star, in units where G=c=Msun=1
    lapse: float, gravitational time dilation at center of neutron star
    '''
    # Create radial grid from 0 to rmax with npoints samples
    r = np.linspace(0.0, rmax, npoints)
    
    # Set up initial conditions at the center (r=0)
    central_press = eos_press_from_rho(rhoc, eos_Gamma, eos_kappa)
    initial_state = (central_press, 0.0, 0.0)  # (pressure, mass, potential)
    
    # Integrate the TOV equations using scipy's odeint
    solution = si.odeint(tov_RHS, initial_state, r, args=(eos_Gamma, eos_kappa))
    
    # Extract individual solution components
    press_profile = solution[:, 0]
    mass_profile = solution[:, 1]
    phi_profile = solution[:, 2]
    
    # Identify the surface of the star (where pressure becomes effectively zero)
    epsilon = 1e-12  # Small threshold to handle floating-point precision
    positive_press_indices = np.where(press_profile > epsilon)[0]
    
    if len(positive_press_indices) == 0:
        # Unphysical case: no positive pressure despite non-zero central density
        return (0.0, 1.0)
    
    last_positive_idx = positive_press_indices[-1]
    
    if last_positive_idx == npoints - 1:
        # Pressure remains positive at rmax (violates problem constraints, handle gracefully)
        surface_radius = rmax
        total_mass = mass_profile[-1]
        phi_at_surface_relative = phi_profile[-1]
    else:
        # Interpolate to find precise surface radius where pressure reaches zero
        r_inner = r[last_positive_idx]
        press_inner = press_profile[last_positive_idx]
        r_outer = r[last_positive_idx + 1]
        press_outer = press_profile[last_positive_idx + 1]
        
        # Linear interpolation for surface radius
        surface_radius = r_inner - press_inner * (r_outer - r_inner) / (press_outer - press_inner)
        
        # Interpolate to get total gravitational mass at the surface
        mass_inner = mass_profile[last_positive_idx]
        mass_outer = mass_profile[last_positive_idx + 1]
        total_mass = mass_inner + (mass_outer - mass_inner) * (surface_radius - r_inner) / (r_outer - r_inner)
        
        # Interpolate to get relative potential at surface (integral of dphi/dr from 0 to surface)
        phi_inner = phi_profile[last_positive_idx]
        phi_outer = phi_profile[last_positive_idx + 1]
        phi_at_surface_relative = phi_inner + (phi_outer - phi_inner) * (surface_radius - r_inner) / (r_outer - r_inner)
    
    # Calculate required potential at surface from outer Schwarzschild boundary condition
    boundary_argument = 1.0 - 2.0 * total_mass / surface_radius
    
    if boundary_argument <= 0.0:
        # Star is a black hole (unphysical for neutron star), set lapse to 0
        central_time_dilation = 0.0
    else:
        # Compute absolute potential at surface
        phi_at_surface_absolute = 0.5 * np.log(boundary_argument)
        # Calculate central potential by matching boundary condition
        central_phi = phi_at_surface_absolute - phi_at_surface_relative
        # Gravitational time dilation is e^phi
        central_time_dilation = np.exp(central_phi)
    
    return (total_mass, central_time_dilation)
