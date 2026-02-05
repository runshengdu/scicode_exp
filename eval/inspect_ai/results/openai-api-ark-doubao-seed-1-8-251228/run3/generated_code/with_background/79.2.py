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
    # Calculate initial force at t
    F0 = -m * omega**2 * x0
    # Update velocity to t + dt/2
    v_half = v0 + (F0 / m) * (dt / 2)
    # Update position to t + dt
    xt = x0 + v_half * dt
    # Calculate force at t + dt
    F_new = -m * omega**2 * xt
    # Update velocity to t + dt
    vt = v_half + (F_new / m) * (dt / 2)
    
    return [vt, xt]



def nhc_step(v0, G, V, X, dt, m, T, omega):
    '''Calculate the position and velocity of the harmonic oscillator using the Nosé-Hoover-chain Liouville operator
    Inputs:
    v0 : float
        The initial velocity of the harmonic oscillator.
    G : float
        The initial force of the harmonic oscillator.
    V : float or array-like
        The initial velocities of the Nosé-Hoover chain variables (v_ξi).
    X : float or array-like
        The initial positions of the Nosé-Hoover chain variables (ξi).
    dt : float
        The integration time step.
    m : float
        The mass of the harmonic oscillator.
    T : float
        The temperature of the harmonic oscillator.
    omega: float
        The frequency of the harmonic oscillator.
    Output:
    v : float
        The updated velocity of the harmonic oscillator.
    G : float
        The updated force of the harmonic oscillator (unchanged in NHC step).
    V : float or array-like
        The updated velocities of the Nosé-Hoover chain variables.
    X : float or array-like
        The updated positions of the Nosé-Hoover chain variables.
    '''
    # Convert inputs to numpy arrays for uniform manipulation
    V = np.atleast_1d(V).astype(float)
    X = np.atleast_1d(X).astype(float)
    M = V.size
    v = v0  # Initialize with initial velocity

    # First phase: update chain variables from M to 1 (1-based indexing)
    if M == 1:
        # Single chain variable case - no cross terms
        G1 = (m * v0**2 - T)
        V[0] += (dt / 4) * G1
    else:
        # Update the highest chain variable (M-th)
        G_M = (V[M-2]**2 - T)
        V[M-1] += (dt / 4) * G_M
        # Apply exponential scaling to (M-1)-th variable
        V[M-2] *= np.exp(-(dt / 8) * V[M-1])

        # Update middle chain variables from M-1 down to 2
        for i in range(M-2, 0, -1):
            G_k = (V[i-1]**2 - T)
            V[i] += (dt / 4) * G_k
            V[i] *= np.exp(-(dt / 8) * V[i+1])

        # Update the first chain variable
        V[0] *= np.exp(-(dt / 8) * V[1])
        G1 = (m * v0**2 - T)
        V[0] += (dt / 4) * G1
        V[0] *= np.exp(-(dt / 8) * V[1])

    # Update oscillator velocity and chain positions
    v = v0 * np.exp(-(dt / 2) * V[0])
    X += (dt / 2) * V

    # Second phase: update chain variables from 1 to M (1-based indexing)
    if M == 1:
        # Single chain variable case
        G1 = (m * v**2 - T)
        V[0] += (dt / 4) * G1
    else:
        # Update the first chain variable
        V[0] *= np.exp(-(dt / 8) * V[1])
        G1 = (m * v**2 - T)
        V[0] += (dt / 4) * G1
        V[0] *= np.exp(-(dt / 8) * V[1])

        # Update middle chain variables from 2 to M-1
        for i in range(1, M-1):
            V[i] *= np.exp(-(dt / 8) * V[i+1])
            G_k = (V[i-1]**2 - T)
            V[i] += (dt / 4) * G_k
            V[i] *= np.exp(-(dt / 8) * V[i+1])

        # Update the highest chain variable (M-th)
        G_M = (V[M-2]**2 - T)
        V[M-1] += (dt / 4) * G_M

    # Convert back to scalar if original input was scalar
    if V.size == 1:
        V = V.item()
    if X.size == 1:
        X = X.item()

    # Return updated values; G remains unchanged as NHC doesn't affect oscillator position
    return v, G, V, X
