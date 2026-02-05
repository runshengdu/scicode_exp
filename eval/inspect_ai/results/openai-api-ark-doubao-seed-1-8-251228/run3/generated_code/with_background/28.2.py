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




def gaussian_beam_through_lens(wavelength, w0, R0, Mf1, z, Mp2, L1, s):
    '''gaussian_beam_through_lens simulates the propagation of a Gaussian beam through an optical lens system
    and calculates the beam waist size at various distances from the lens.
    Input
    - wavelength: float, The wavelength of the light (in meters).
    - w0: float, The waist radius of the Gaussian beam before entering the optical system (in meters).
    - R0: float, The radius of curvature of the beam's wavefront at the input plane (in meters).
    - Mf1: The ABCD matrix representing the first lens or optical element in the system, 2D numpy array with shape (2,2)
    - z: A 1d numpy array of float distances (in meters) from the lens where the waist size is to be calculated.
    - Mp2: float, A scaling factor used in the initial complex beam parameter calculation.
    - L1: float, The distance (in meters) from the first lens or optical element to the second one in the optical system.
    - s: float, the distance (in meters) from the source (or initial beam waist) to the first optical element.
    Output
    wz: waist over z, a 1d array of float, same shape of z
    '''
    # Calculate initial complex beam parameter at input plane (source plane)
    inv_q_input = 1.0 / R0 - 1j * (wavelength * Mp2) / (np.pi * w0 ** 2)
    q_input = 1.0 / inv_q_input
    
    # Propagate from input plane to first lens (free space propagation over distance s)
    # ABCD matrix for free space: [[1, s], [0, 1]]
    q_lens_input = q_input + s
    
    # Apply ABCD matrix of the first lens to get complex parameter after first lens
    A, B = Mf1[0, 0], Mf1[0, 1]
    C, D = Mf1[1, 0], Mf1[1, 1]
    q_after_first_lens = (A * q_lens_input + B) / (C * q_lens_input + D)
    
    # Initialize output array for waist/spot sizes
    wz = np.full_like(z, np.nan, dtype=np.float64)
    
    for idx, z_val in enumerate(z):
        if z_val < L1:
            # Propagate from first lens to current distance (before reaching second element)
            q_current = q_after_first_lens + z_val
        else:
            # Propagate from first lens to second element (free space over L1)
            q_before_second = q_after_first_lens + L1
            # Assume second element is a propagation step (scaling factor Mp2 is applied in final formula)
            # Propagate from second element to current distance
            q_current = q_before_second + (z_val - L1)
        
        # Calculate reciprocal of complex parameter
        inv_q_current = 1.0 / q_current
        
        # Extract imaginary part and compute spot size using given formula
        imag_inv_q = np.imag(inv_q_current)
        if imag_inv_q < 0:  # Ensure valid input for square root
            wz[idx] = np.sqrt((-wavelength * Mp2) / (np.pi * imag_inv_q))
    
    return wz
