import numpy as np


def calculate_paraxial(h1, r1, r2, r3, d1, d2, n1, n2, n3, n_total):
    '''Computes the axial parameters for spherical aberration calculation.
    Parameters:
    - h1 (array of floats): Aperture heights, in range (0.01, hm)
    - r1 (float): Radius of curvature for the first lens surface
    - r2 (float): Radius of curvature for the second lens surface
    - r3 (float): Radius of curvature for the third lens surface
    - d1 (float): Separation distance between first and second surfaces
    - d2 (float): Separation distance between second and third surfaces
    - n1 (float): Refractive index of the first lens material
    - n2 (float): Refractive index of the second lens material
    - n3 (float): Refractive index of the third lens material
    - n_total (float): Refractive index of the surrounding medium (air)
    Returns:
    - l31 (array of floats): Axial image locations for the third surface
    '''
    
    # Process first surface (incident from air to first lens)
    n_in = n_total
    n_out = n1
    r = r1
    # Object at infinity, so calculate i directly from incident height
    i = h1 / r
    i_prime = (n_in / n_out) * i
    u_prime = 0.0 + i - i_prime  # Initial u is 0 (parallel rays)
    l_prime = (r * i) / u_prime + r
    l1_prime = l_prime
    u1_prime = u_prime
    
    # Transition to second surface
    l2 = l1_prime - d1
    u2 = u1_prime
    n_in_2 = n_out  # n for second surface is output of first surface
    
    # Process second surface (first lens to second lens)
    n_in = n_in_2
    n_out = n2
    r = r2
    l = l2
    u = u2
    i = (l - r) / r * u
    i_prime = (n_in / n_out) * i
    u_prime = u + i - i_prime
    l_prime = (r * i) / u_prime + r
    l2_prime = l_prime
    u2_prime = u_prime
    
    # Transition to third surface
    l3 = l2_prime - d2
    u3 = u2_prime
    n_in_3 = n_out  # n for third surface is output of second surface
    
    # Process third surface (second lens to third lens material)
    n_in = n_in_3
    n_out = n3
    r = r3
    l = l3
    u = u3
    i = (l - r) / r * u
    i_prime = (n_in / n_out) * i
    u_prime = u + i - i_prime
    l_prime = (r * i) / u_prime + r
    l3_prime = l_prime
    
    # Return axial image locations using third lens as origin
    return l3_prime



def calculate_non_paraxial(h1, r1, r2, r3, d1, d2, n1, n2, n3, n_total):
    '''Computes the paraxial parameters for spherical aberration calculation.
    Parameters:
    - h1 (array of floats): Aperture heights, in range (0.01, hm)
    - r1 (float): Radius of curvature for the first lens surface
    - r2 (float): Radius of curvature for the second lens surface
    - r3 (float): Radius of curvature for the third lens surface
    - d1 (float): Separation distance between first and second surfaces
    - d2 (float): Separation distance between second and third surfaces
    - n1 (float): Refractive index of the first lens material
    - n2 (float): Refractive index of the second lens material
    - n3 (float): Refractive index of the third lens material
    - n_total (float): Refractive index of the surrounding medium (air)
    Returns:
    - L31 (array of floats): Paraxial image locations for the third surface
    '''
    # Process first surface (incident from air to first lens)
    n_in = n_total
    n_out = n1
    r = r1
    U1 = 0.0  # Parallel rays from infinity
    
    sin_I1 = h1 / r
    I1 = np.arcsin(sin_I1)
    sin_I1_prime = (n_in / n_out) * sin_I1
    I1_prime = np.arcsin(sin_I1_prime)
    U1_prime = U1 + I1 - I1_prime
    L1_prime = r + r * (sin_I1_prime / np.sin(U1_prime))
    
    # Transition to second surface
    L2 = L1_prime - d1
    U2 = U1_prime
    n_in_2 = n_out  # n for second surface is output of first surface
    
    # Process second surface (first lens to second lens)
    n_in = n_in_2
    n_out = n2
    r = r2
    L = L2
    U = U2
    
    sin_I2 = (L - r) / r * np.sin(U)
    I2 = np.arcsin(sin_I2)
    sin_I2_prime = (n_in / n_out) * sin_I2
    I2_prime = np.arcsin(sin_I2_prime)
    U2_prime = U2 + I2 - I2_prime
    L2_prime = r + r * (sin_I2_prime / np.sin(U2_prime))
    
    # Transition to third surface
    L3 = L2_prime - d2
    U3 = U2_prime
    n_in_3 = n_out  # n for third surface is output of second surface
    
    # Process third surface (second lens to third lens material)
    n_in = n_in_3
    n_out = n3
    r = r3
    L = L3
    U = U3
    
    sin_I3 = (L - r) / r * np.sin(U)
    I3 = np.arcsin(sin_I3)
    sin_I3_prime = (n_in / n_out) * sin_I3
    I3_prime = np.arcsin(sin_I3_prime)
    U3_prime = U3 + I3 - I3_prime
    L3_prime = r + r * (sin_I3_prime / np.sin(U3_prime))
    
    L31 = L3_prime
    return L31
