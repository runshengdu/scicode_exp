import numpy as np



def harmonic_mannella_leapfrog(x0, v0, t0, steps, taup, omega0, vrms):
    '''Function to employ Mannella's leapfrog method to solve the Langevin equation of a microsphere optically trapped in the gas.
    Input
    x0 : float
        Initial position of the microsphere.
    v0 : float
        Initial velocity of the microsphere.
    t0 : float
        Total simulation time.
    steps : int
        Number of integration steps.
    taup : float
        Momentum relaxation time of the trapped microsphere in the gas (often referred to as the particle relaxation time).
    omega0 : float
        Resonant frequency of the harmonic potential (optical trap).
    vrms : float
        Root mean square velocity of the trapped microsphere in the gas.
    Output
    x : float
        Final position of the microsphere after the simulation time.
    '''
    delta_t = t0 / steps
    x = x0
    v = v0
    
    for _ in range(steps):
        # Compute mid-step position
        x_half = x + v * delta_t / 2
        
        # Generate Wiener increment for stochastic term
        delta_W = np.random.normal(0, np.sqrt(delta_t))
        
        # Calculate numerator and denominator for updated velocity
        numerator = (v - v * delta_t / (2 * taup) - 
                    (omega0 ** 2) * x_half * delta_t + 
                    np.sqrt(2 / taup) * vrms * delta_W)
        denominator = 1 + delta_t / (2 * taup)
        v_next = numerator / denominator
        
        # Compute next full-step position
        x_next = x_half + v_next * delta_t / 2
        
        # Update state variables for next iteration
        x, v = x_next, v_next
    
    return x
