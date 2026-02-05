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



def analyze_structure_factor(Sk, Kx, Ky, q0, min_height):
    '''Analyze the structure factor to identify peak near q0
    Input:
    Sk: 2D structure factor, 2D array of floats
    Kx: coordinates in k space conjugate to x in real space, 2D N*N array of floats
    Ky: coordinates in k space conjugate to y in real space, 2D N*N array of floats
    q0: critical mode, float
    min_height: threshold height of the peak in the structure factor to be considered, float
    Output:
    peak_found_near_q0: if a peak is found in the structure factor near q0 (meets combined proximity, narrowness, and height criteria), boolean
    peak_near_q0_location: location of the peak near q0 in terms of radial k, float; set to 0 if no peak is found near q0
    '''
    N = Sk.shape[0]
    # Compute radial wavevector magnitude for each k-space point
    kr = np.sqrt(Kx**2 + Ky**2)
    
    # Flatten arrays to handle radial binning
    kr_flat = kr.flatten()
    Sk_flat = Sk.flatten()
    
    # Compute unique radial wavevectors and radial average of structure factor
    unique_kr, inverse_indices = np.unique(kr_flat, return_inverse=True)
    counts = np.bincount(inverse_indices)
    sum_Sk = np.bincount(inverse_indices, weights=Sk_flat)
    radial_avg = sum_Sk / counts  # Average Sk values for each radial shell
    
    # Find local maxima (peaks) in radial average with height above threshold
    peaks_indices, peaks_properties = find_peaks(radial_avg, height=min_height)
    if len(peaks_indices) == 0:
        return False, 0.0
    
    peak_kr = unique_kr[peaks_indices]
    peak_heights = peaks_properties['peak_heights']
    n_radial = len(radial_avg)
    
    # Calculate FWHM (Full Width at Half Maximum) for each peak to assess narrowness
    fwhms = []
    for idx in peaks_indices:
        peak_h = radial_avg[idx]
        half_h = 0.5 * peak_h
        
        # Find left boundary of FWHM (first point left of peak with value <= half height)
        left_idx = idx - 1
        while left_idx >= 0 and radial_avg[left_idx] > half_h:
            left_idx -= 1
        
        if left_idx < 0:
            left_kr_val = 0.0
        else:
            x0, y0 = unique_kr[left_idx], radial_avg[left_idx]
            x1, y1 = unique_kr[left_idx + 1], radial_avg[left_idx + 1]
            if np.isclose(y1, y0):
                x_left = x1
            else:
                x_left = x0 + (half_h - y0) * (x1 - x0) / (y1 - y0)
            left_kr_val = x_left
        
        # Find right boundary of FWHM (first point right of peak with value <= half height)
        right_idx = idx + 1
        while right_idx < n_radial and radial_avg[right_idx] > half_h:
            right_idx += 1
        
        if right_idx >= n_radial:
            right_kr_val = unique_kr[-1]
        else:
            x0, y0 = unique_kr[right_idx - 1], radial_avg[right_idx - 1]
            x1, y1 = unique_kr[right_idx], radial_avg[right_idx]
            if np.isclose(y1, y0):
                x_right = x0
            else:
                x_right = x0 + (half_h - y0) * (x1 - x0) / (y1 - y0)
            right_kr_val = x_right
        
        fwhms.append(right_kr_val - left_kr_val)
    
    fwhms = np.array(fwhms)
    
    # Define tolerance criteria for proximity and narrowness
    dk = 2 * np.pi / N  # Smallest k-space increment in x/y direction
    proximity_tolerance = max(0.1 * q0, dk)  # 10% relative or 1 bin width, whichever is larger
    narrowness_tolerance = max(0.2 * q0, 2 * dk)  # 20% relative or 2 bin widths, whichever is larger
    
    # Filter peaks that meet both proximity and narrowness criteria
    distances = np.abs(peak_kr - q0)
    valid_mask = (distances <= proximity_tolerance) & (fwhms <= narrowness_tolerance)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return False, 0.0
    
    # Select the best peak: closest to q0, then highest if ties exist
    valid_peak_kr = peak_kr[valid_indices]
    valid_distances = distances[valid_indices]
    valid_heights = peak_heights[valid_indices]
    
    min_dist = np.min(valid_distances)
    min_dist_peaks = valid_indices[valid_distances == min_dist]
    
    if len(min_dist_peaks) == 1:
        best_peak_idx = min_dist_peaks[0]
    else:
        # Choose highest peak among those with minimal distance to q0
        best_peak_idx = min_dist_peaks[np.argmax(valid_heights[valid_distances == min_dist])]
    
    best_peak_kr = peak_kr[best_peak_idx]
    
    return True, best_peak_kr
