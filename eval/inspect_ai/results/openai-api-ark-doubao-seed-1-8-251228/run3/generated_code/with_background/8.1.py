import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift



def apply_cshband_pass_filter(image_array, bandwidth):
    '''Applies a cross shaped high band pass filter to the given image array based on the frequency bandwidth.
    Input:
    image_array: float;2D numpy array, the input image.
    bandwitdh: bandwidth of cross-shaped filter, int
    Ouput:
    T: 2D numpy array of float, The spatial filter used.
    filtered_image: 2D numpy array of float, the filtered image in the original domain.
    '''
    m, n = image_array.shape
    
    # Create centered frequency coordinate grids
    u_grid = np.arange(m)[:, np.newaxis] - m // 2
    v_grid = np.arange(n)[np.newaxis, :] - n // 2
    
    abs_u = np.abs(u_grid)
    abs_v = np.abs(v_grid)
    
    # Define cross-shaped high band pass filter mask (exclude bandwidth frequency)
    condition_horizontal = (abs_u < bandwidth) & (abs_v > bandwidth)
    condition_vertical = (abs_v < bandwidth) & (abs_u > bandwidth)
    T = np.where(condition_horizontal | condition_vertical, 1.0, 0.0)
    
    # Fourier transform pipeline
    # Forward FFT and shift DC to center
    fft_shifted = fftshift(fft2(image_array))
    # Apply filter in frequency domain
    fft_filtered_shifted = fft_shifted * T
    # Shift back and inverse FFT
    filtered_image_complex = ifft2(ifftshift(fft_filtered_shifted))
    # Extract real part as input image is real-valued
    filtered_image = np.real(filtered_image_complex)
    
    return T, filtered_image
