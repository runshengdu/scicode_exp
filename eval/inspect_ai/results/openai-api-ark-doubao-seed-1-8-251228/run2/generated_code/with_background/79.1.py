import numpy as np



def Verlet(v0, x0, m, dt, omega):
    '''Calculate the position and velocity of the harmonic oscillator using the velocity-Verlet algorithm
    Inputs:
    v0 : float
        The initial velocity of the harmonic oscillator.
    x0 : float
        The initial position of the harmonic oscillator.
    m : float
    dt : float
        The integration time step.
    omega: float
    Output:
    [vt, xt] : list
        The updated velocity and position of the harmonic oscillator.
    '''
    # Calculate force at time t
    F_t = -m * omega**2 * x0
    # Update velocity to t + dt/2
    v_half = v0 + (F_t / m) * (dt / 2)
    # Update position to t + dt
    xt = x0 + v_half * dt
    # Calculate force at time t + dt
    F_t_plus_dt = -m * omega**2 * xt
    # Update velocity to t + dt
    vt = v_half + (F_t_plus_dt / m) * (dt / 2)
    
    return [vt, xt]
