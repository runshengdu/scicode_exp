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
