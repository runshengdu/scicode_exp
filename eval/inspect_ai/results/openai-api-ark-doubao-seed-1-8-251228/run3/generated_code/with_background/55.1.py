import numpy as np
from numpy.fft import fft2, ifft2, fftshift, rfft2, irfft2, fftfreq, rfftfreq
from scipy.signal import find_peaks, peak_widths



def solve_SH(u, dt, T, N, epsilon, q0):
    '''Run a 2D simulation of Swift-Hohenberg equation
    Input
    u: initial condition of the order parameter, 2D array of floats
    dt: time step size, float
    T: total time of the ecolution, float
    N: system size where the 2D system is of dimension N*N, int
    epsilon: control parameter, float
    q0: critical mode, float
    Output
    u: final state of the system at the end time T.
    '''
    current_u = u.copy()
    current_time = 0.0
    
    # Precompute k-space grids and squared wave numbers
    kx = 2 * np.pi * np.fft.fftfreq(N)
    ky = 2 * np.pi * np.fft.fftfreq(N)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    k_squared = kx_grid ** 2 + ky_grid ** 2
    
    while current_time < T:
        # Adjust time step for final iteration if needed
        step_dt = min(dt, T - current_time)
        
        # Step 1: Update nonlinear and local linear term in real space
        reaction_term = (epsilon - 1) * current_u - current_u ** 3
        u1 = current_u + step_dt * reaction_term
        
        # Step 2: Update linear derivative terms in k-space
        # Transform intermediate state to k-space
        u1_hat = np.fft.fft2(u1)
        
        # Compute exponential factor for k-space evolution
        linear_operator_k = 2 * k_squared / (q0 ** 2) - (k_squared ** 2) / (q0 ** 4)
        exp_factor = np.exp(step_dt * linear_operator_k)
        
        # Evolve in k-space and transform back to real space
        u_next_hat = u1_hat * exp_factor
        current_u = np.fft.ifft2(u_next_hat).real
        
        # Update elapsed time
        current_time += step_dt
    
    return current_u
