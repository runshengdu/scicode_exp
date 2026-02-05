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


def nhc_Y4(v0, G, V, X, dt, m, T, omega):
    '''Use the Yoshida's fourth-order method to give a more acurate evolution of the extra variables
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
        The mass of the harmonic oscillator.
    T : float
        The temperature of the harmonic oscillator.
    omega : float
        The frequency of the harmonic oscillator.
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
    # Calculate Yoshida fourth-order weights
    w1 = 1.0 / (2 - np.cbrt(2))
    w2 = 1.0 - 2 * w1
    
    # First Yoshida step with weight w1
    v, G, V, X = nhc_step(v0, G, V, X, w1 * dt, m, T, omega)
    
    # Second Yoshida step with weight w2
    v, G, V, X = nhc_step(v, G, V, X, w2 * dt, m, T, omega)
    
    # Third Yoshida step with weight w1
    v, G, V, X = nhc_step(v, G, V, X, w1 * dt, m, T, omega)
    
    return v, G, V, X




def nose_hoover_chain(x0, v0, T, M, m, omega, dt, nsteps):
    '''Integrate the full Liouville operator of the Nose-Hoover-chain thermostat and get the trajectories of the harmonic oscillator
    Inputs:
    x0 : float
        The initial position of the harmonic oscillator.
    v0 : float
        The initial velocity of the harmonic oscillator.
    T : float
        The temperature of the harmonic oscillator.
    M : int
        The number of Nose-Hoover-chains.
    m : float
        The mass of the harmonic oscillator.
    omega : float
        The frequency of the harmonic oscillator.
    dt : float
        The integration time step.
    nsteps : int
        The number of integration time steps.
    Outputs:
    x : array of shape (nsteps, 1)
        The position trajectory of the harmonic oscillator.
    v : array of shape (nsteps, 1)
        The velocity trajectory of the harmonic oscillator.
    '''
    # Initialize trajectory arrays
    x_trajectory = np.zeros((nsteps, 1))
    v_trajectory = np.zeros((nsteps, 1))
    
    # Current state variables
    current_x = x0
    current_v = v0
    # Initialize NHC chain variables: v_xi (V) and xi (X)
    V = np.zeros(M)
    X = np.zeros(M)
    
    # Yoshida fourth-order weights
    w1 = 1.0 / (2 - np.cbrt(2))
    w2 = 1.0 - 2 * w1
    
    def apply_nhc_step(v_initial, V_initial, X_initial, dt_step, mass, temp, num_chains):
        # Helper function to perform a single NHC integration step
        V_copy = V_initial.copy()
        X_copy = X_initial.copy()
        v = v_initial
        
        # First phase: update chain velocities from M down to 1 (1-based indexing)
        if num_chains >= 1:
            if num_chains == 1:
                # Single chain variable case
                G = mass * v_initial ** 2 - temp
                V_copy[0] += (dt_step / 4) * G
            else:
                # Update M-th chain variable (1-based, index num_chains-1 in 0-based)
                G = V_copy[num_chains - 2] ** 2 - temp
                V_copy[num_chains - 1] += (dt_step / 4) * G
                
                # Update from M-1 down to 2 (1-based)
                for i in range(num_chains - 1, 1, -1):
                    # i is 1-based, convert to 0-based index
                    idx = i - 1
                    V_copy[idx] *= np.exp(-(dt_step / 8) * V_copy[idx + 1])
                    G = V_copy[idx - 1] ** 2 - temp
                    V_copy[idx] += (dt_step / 4) * G
                    V_copy[idx] *= np.exp(-(dt_step / 8) * V_copy[idx + 1])
                
                # Update first chain variable (1-based, index 0 in 0-based)
                V_copy[0] *= np.exp(-(dt_step / 8) * V_copy[1])
                G = mass * v_initial ** 2 - temp
                V_copy[0] += (dt_step / 4) * G
                V_copy[0] *= np.exp(-(dt_step / 8) * V_copy[1])
        
        # Update oscillator velocity and chain positions
        v = v_initial * np.exp(-(dt_step / 2) * V_copy[0])
        X_copy += (dt_step / 2) * V_copy
        
        # Final phase: update chain velocities from 1 to M (1-based indexing)