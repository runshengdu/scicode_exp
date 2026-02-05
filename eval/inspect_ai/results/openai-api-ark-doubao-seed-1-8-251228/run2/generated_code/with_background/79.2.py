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



def nhc_step(v0, G, V, X, dt, m, T, omega):
    '''Calculate the position and velocity of the harmonic oscillator using the Nosé-Hoover-chain Liouville operator
    Inputs:
    v0 : float
        The initial velocity of the harmonic oscillator.
    G : float
        The initial force of the harmonic oscillator.
    V : float
        The initial velocity of the particles.
    X : float
        The initial position of the particles.
    dt : float
        The integration time step.
    m : float
    T : float
        The temperature of the harmonic oscillator.
    omega: float
    Output:
    v : float
        The updated velocity of the harmonic oscillator.
    G : float
        The updated force of the harmonic oscillator.
    V : float
        The updated velocity of the particles.
    X : float
        The updated position of the particles.
    '''
    # Assume M=1 (single NHC chain variable), Q1=1, and k_B=1
    # First phase: update chain velocity with initial G1
    G1_initial = m * v0 ** 2 - T
    V += (dt / 4) * G1_initial
    
    # Update oscillator velocity and chain position
    v = v0 * np.exp(-(dt / 2) * V)
    X += (dt / 2) * V
    
    # Final phase: update chain velocity with new G1 (using updated oscillator velocity)
    G1_final = m * v ** 2 - T
    V += (dt / 4) * G1_final
    
    # Oscillator force remains unchanged as position x is not modified in this step
    return v, G, V, X
