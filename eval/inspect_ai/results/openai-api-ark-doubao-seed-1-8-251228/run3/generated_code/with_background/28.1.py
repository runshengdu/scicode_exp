import numpy as np
from scipy.integrate import simps




def propagate_gaussian_beam(N, Ld, w0, z, L):
    '''Propagate a Gaussian beam and calculate its intensity distribution before and after propagation.
    Input
    N : int
        The number of sampling points in each dimension (assumes a square grid).
    Ld : float
        Wavelength of the Gaussian beam.
    w0 : float
        Waist radius of the Gaussian beam at its narrowest point.
    z : float
        Propagation distance of the Gaussian beam.
    L : float
        Side length of the square area over which the beam is sampled.
    Output
    Gau:     a 2D array with dimensions (N+1, N+1) representing the absolute value of the beam's amplitude distribution before propagation.
    Gau_Pro: a 2D array with dimensions (N+1, N+1) representing the absolute value of the beam's amplitude distribution after propagation.
    '''
    # Create coordinate grid for the initial plane (z=0)
    N_plus_1 = N + 1
    x = np.linspace(-L/2, L/2, N_plus_1)
    y = np.linspace(-L/2, L/2, N_plus_1)
    X, Y = np.meshgrid(x, y)
    
    # Generate initial Gaussian beam field at z=0 (complex field, purely real in this case)
    E0 = (1 / w0) * np.exp(-(X**2 + Y**2) / w0**2)
    
    # Compute 2D FFT of the initial field and shift zero frequency to center
    E0_fft = np.fft.fft2(E0)
    E0_fft_shifted = np.fft.fftshift(E0_fft)
    
    # Calculate spatial frequency (wavenumber) grids
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    kx = 2 * np.pi * np.fft.fftfreq(N_plus_1, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(N_plus_1, d=dy)
    kx_shifted = np.fft.fftshift(kx)
    ky_shifted = np.fft.fftshift(ky)
    KX, KY = np.meshgrid(kx_shifted, ky_shifted)
    
    # Compute propagation kernel in frequency domain
    k = 2 * np.pi / Ld  # Wavenumber of the Gaussian beam
    # Calculate sqrt(k² - kx² - ky²) with complex handling for evanescent waves
    sqrt_term = np.sqrt(k**2 - KX**2 - KY**2 + 0j)
    propagation_kernel = np.exp(1j * z * sqrt_term)
    
    # Apply propagation kernel and inverse transform back to spatial domain
    E_fft_shifted = E0_fft_shifted * propagation_kernel
    E_fft = np.fft.ifftshift(E_fft_shifted)
    E_z = np.fft.ifft2(E_fft)
    
    # Calculate absolute amplitude distributions
    Gau = np.abs(E0)
    Gau_Pro = np.abs(E_z)
    
    return Gau, Gau_Pro
