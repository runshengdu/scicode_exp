import numpy as np
from numpy.fft import fft2, ifft2, fftshift, rfft2, irfft2, fftfreq, rfftfreq
from scipy.signal import find_peaks, peak_widths

def solve_SH(u, dt, T, N, epsilon, q0):
    '''Run a 2D simulation of Swift-Hohenberg equation
    Input
    u: initial condition of the order parameter, 2D array of floats
    dt: time step size, float
    T: total time of the evolution, float
    N: system size where the 2D system is of dimension N*N, int
    epsilon: control parameter, float
    q0: critical mode, float
    Output
    u: final state of the system at the end time T.
    '''
    # Initialize current state with high precision
    u_current = u.copy().astype(np.float64)
    current_time = 0.0
    
    # Precompute wavevectors and linear operator in k-space
    kx = 2 * np.pi * np.fft.fftfreq(N)
    ky = 2 * np.pi * np.fft.fftfreq(N)
    kx_grid = kx[:, np.newaxis]  # Shape (N, 1) for broadcasting
    ky_grid = ky[np.newaxis, :]  # Shape (1, N) for broadcasting
    k_squared = kx_grid ** 2 + ky_grid ** 2
    # Compute linear operator symbol in k-space
    L_hat = 2 * k_squared / (q0 ** 2) - (k_squared ** 2) / (q0 ** 4)
    
    a = epsilon - 1.0  # Coefficient for the nonlinear ODE
    
    while current_time < T:
        # Handle final time step if T is not a multiple of dt
        delta_t = min(dt, T - current_time)
        
        # Step 1: Exact integration of nonlinear term in real space
        if abs(a) < 1e-12:
            # Special case: epsilon = 1, du/dt = -u^3
            denominator = 1.0 + 2 * delta_t * u_current ** 2
            u1 = u_current / np.sqrt(denominator)
        else:
            # General case: solve du/dt = a*u - u^3 using exact separable solution
            exp_2a_dt = np.exp(2 * a * delta_t)
            numerator = a * (u_current ** 2) * exp_2a_dt
            denominator = a + u_current ** 2 * (exp_2a_dt - 1)
            u1_squared = numerator / denominator
            # Preserve sign of original u_current
            u1 = np.sign(u_current) * np.sqrt(u1_squared)
        
        # Step 2: Linear term evolution in k-space
        # Fourier transform intermediate state to k-space
        u1_hat = np.fft.fft2(u1)
        # Apply exponential time evolution factor
        exp_factor = np.exp(delta_t * L_hat)
        u_next_hat = u1_hat * exp_factor
        # Inverse transform back to real space
        u_next = np.fft.ifft2(u_next_hat)
        # Discard spurious imaginary parts from numerical errors
        u_current = np.real(u_next)
        
        # Update time counter
        current_time += delta_t
    
    return u_current



def structure_factor(u):
    '''Calculate the structure factor of a 2D real spatial distribution and the Fourier coordinates, shifted to center around k = 0
    Input
    u: order parameter in real space, 2D N*N array of floats
    Output
    Kx: coordinates in k space conjugate to x in real space, 2D N*N array of floats
    Ky: coordinates in k space conjugate to y in real space, 2D N*N array of floats
    Sk: 2D structure factor, 2D array of floats
    '''
    N = u.shape[0]
    # Compute Fourier transform of the real-space field
    U = fft2(u)
    # Calculate structure factor and shift to center k=0
    Sk = np.fft.fftshift(np.abs(U) ** 2)
    # Compute 1D k-space coordinates
    kx = 2 * np.pi * np.fft.fftfreq(N)
    ky = 2 * np.pi * np.fft.fftfreq(N)
    # Shift coordinates to center zero frequency
    kx_shifted = np.fft.fftshift(kx)
    ky_shifted = np.fft.fftshift(ky)
    # Create 2D coordinate grids aligned with shifted k-space
    Kx, Ky = np.meshgrid(kx_shifted, ky_shifted, indexing='ij')
    return Kx, Ky, Sk
