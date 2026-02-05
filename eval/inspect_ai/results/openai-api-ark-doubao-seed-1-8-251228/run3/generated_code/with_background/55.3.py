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
    # Compute k-space wave numbers
    kx = 2 * np.pi * np.fft.fftfreq(N)
    ky = 2 * np.pi * np.fft.fftfreq(N)
    # Create 2D k-space grids
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    # Shift grids to center k=0
    Kx = np.fft.fftshift(kx_grid)
    Ky = np.fft.fftshift(ky_grid)
    # Compute Fourier transform of u
    U = np.fft.fft2(u)
    # Calculate structure factor and shift to center
    Sk = np.fft.fftshift(np.abs(U) ** 2)
    
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
    # Compute radial wave number grid
    kr_grid = np.sqrt(Kx ** 2 + Ky ** 2)
    kr_flat = kr_grid.ravel()
    sk_flat = Sk.ravel()
    
    # Get system size and fundamental wavevector spacing
    N = Kx.shape[0]
    delta_k = 2 * np.pi / N
    
    # Determine bin edges for radial averaging
    max_kr = np.max(kr_flat)
    bin_edges = np.arange(-delta_k / 2, max_kr + delta_k, delta_k)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Compute counts and sums for each radial bin
    counts, _ = np.histogram(kr_flat, bins=bin_edges)
    sk_sums, _ = np.histogram(kr_flat, bins=bin_edges, weights=sk_flat)
    
    # Calculate radially averaged structure factor (handle division by zero)
    sk_radial = np.where(counts > 0, sk_sums / counts, np.nan)
    
    # Interpolate missing values to create uniformly spaced radial profile
    valid_mask = ~np.isnan(sk_radial)
    if np.any(valid_mask):
        sk_radial_interp = np.interp(bin_centers, bin_centers[valid_mask], sk_radial[valid_mask])
        # Set values beyond the maximum valid kr to 0
        max_valid_kr = bin_centers[valid_mask].max()
        sk_radial_interp[bin_centers > max_valid_kr] = 0.0
    else:
        # This case should not occur as k=0 mode always has count 1
        sk_radial_interp = np.zeros_like(bin_centers)
    
    # Exclude k=0 bin from peak detection (stripe phase peaks are at non-zero k)
    non_zero_mask = bin_centers > 0
    sk_radial_non_zero = sk_radial_interp[non_zero_mask]
    bin_centers_non_zero = bin_centers[non_zero_mask]
    
    if len(sk_radial_non_zero) == 0:
        return False, 0.0
    
    # Detect peaks meeting the minimum height criterion in non-zero k region
    peaks, peak_properties = find_peaks(sk_radial_non_zero, height=min_height)
    
    if len(peaks) == 0:
        return False, 0.0
    
    # Extract peak properties
    peak_krs = bin_centers_non_zero[peaks]
    peak_diffs = np.abs(peak_krs - q0)
    
    # Define criteria thresholds
    proximity_tolerance = max(0.1 * q0, delta_k)
    narrowness_threshold = max(0.2 * q0, 2 * delta_k)
    
    # Calculate peak widths at half height (convert from indices to kr units)
    widths, _, _, _ = peak_widths(sk_radial_non_zero, peaks, rel_height=0.5)
    peak_widths_kr = widths * delta_k
    
    # Filter peaks that meet both proximity and narrowness criteria
    valid_peaks_mask = (peak_diffs < proximity_tolerance) & (peak_widths_kr < narrowness_threshold)
    valid_peak_krs = peak_krs[valid_peaks_mask]
    
    if len(valid_peak_krs) == 0:
        return False, 0.0
    else:
        # Select the peak closest to q0
        closest_idx = np.argmin(peak_diffs[valid_peaks_mask])
        closest_peak_kr = valid_peak_krs[closest_idx]
        return True, closest_peak_kr
