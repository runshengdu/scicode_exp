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



def tov(rhoc, eos_Gamma, eos_kappa, npoints, rmax):
    '''This function computes gravitational time dilation at the center of the neutron star described by a polytropic equation of state as well as the star's mass.
    Inputs
    rhoc: float, the density at the center of the star, in units where G=c=Msun=1.
    eos_Gamma: float, adiabatic exponent of the equation of state
    eos_kappa: float, coefficient of the equation of state
    npoints: int, number of intergration points to use
    rmax: float, maximum radius to which to intgrate solution to, must include the whole star
    Outputs
    mass: float, gravitational mass of neutron star, in units where G=c=Msun=1
    lapse: float, gravitational time dilation at center of neutron star
    '''
    # Compute initial central pressure using polytropic EOS
    press0 = eos_press_from_rho(rhoc, eos_Gamma, eos_kappa)
    initial_data = [press0, 0.0, 0.0]  # [pressure, mass, gravitational potential]

    # Define ODE function wrapping the TOV right-hand side
    def ode_func(r, data):
        return np.array(tov_RHS(tuple(data), r, eos_Gamma, eos_kappa))

    # Define terminal event to stop integration when pressure reaches zero (star surface)
    def surface_event(r, data):
        return data[0]  # Track pressure for zero-crossing
    surface_event.terminal = True
    surface_event.direction = -1  # Trigger only when pressure decreases through zero

    # Generate radial evaluation points
    r_arr = np.linspace(0.0, rmax, npoints)

    # Solve the TOV initial value problem
    sol = si.solve_ivp(
        fun=ode_func,
        t_span=(0.0, rmax),
        y0=initial_data,
        t_eval=r_arr,
        events=[surface_event]
    )

    # Validate that the star surface was reached within rmax
    if not sol.t_events[0].size:
        raise ValueError("rmax is insufficient to contain the star's surface (pressure never reaches zero)")

    # Extract surface properties from the event data
    R = sol.t_events[0][0]
    _, star_mass, phi_R_computed = sol.y_events[0][0]

    # Validate the star is not a black hole (2M/R < 1)
    if 1 - 2 * star_mass / R <= 0:
        raise ValueError("Configuration results in a black hole (2M/R ≥ 1), not a neutron star")

    # Compute the required gravitational potential at the surface from Birkhoff's theorem
    phi_R_true = 0.5 * np.log(1 - 2 * star_mass / R)

    # Calculate central gravitational potential by matching boundary conditions
    phi_c = phi_R_true - phi_R_computed

    # Compute gravitational time dilation (lapse) at the center
    star_lapse = np.exp(phi_c)

    return (star_mass, star_lapse)
