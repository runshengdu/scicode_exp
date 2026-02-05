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
    # Number of points per dimension (output array size)
    M = N + 1
    # Spatial sampling interval
    dx = L / N
    # Create centered coordinate grids
    x = np.linspace(-L/2, L/2, M)
    y = np.linspace(-L/2, L/2, M)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # Initial field at z=0 (normalized to peak amplitude 1)
    E0 = np.exp(-(X**2 + Y**2) / w0**2)
    Gau = np.abs(E0)
    
    # Wave number
    k = 2 * np.pi / Ld
    # Calculate spatial angular frequencies
    kx = 2 * np.pi * np.fft.fftfreq(M, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(M, d=dx)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    
    # Paraxial free-space transfer function
    H = np.exp(1j * z * (k - (KX**2 + KY**2) / (2 * k)))
    
    # Fourier transform of initial field
    E0_hat = np.fft.fft2(E0)
    # Apply propagation transfer function
    E_hat = E0_hat * H
    # Inverse Fourier transform to get propagated field
    E_z = np.fft.ifft2(E_hat)
    
    # Absolute value of propagated amplitude
    Gau_Pro = np.abs(E_z)
    
    return Gau, Gau_Pro
