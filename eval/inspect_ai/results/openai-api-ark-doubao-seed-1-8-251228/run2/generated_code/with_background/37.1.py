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

    h1 = np.asarray(h1)
    
    # Ray tracing through first surface (object at infinity, parallel rays)
    i1 = h1 / r1
    i1_prime = (n_total / n1) * i1
    u1_prime = ((n1 - n_total) * h1) / (n1 * r1)
    l1_prime = (n1 * r1) / (n1 - n_total) + r1 if n1 != n_total else np.inf
    
    # Transition to second surface
    l2 = l1_prime - d1
    u2 = u1_prime
    
    # Ray tracing through second surface
    i2 = (l2 - r2) / r2 * u2 if r2 != 0 else 0
    i2_prime = (n1 / n2) * i2
    u2_prime = u2 + i2 - i2_prime
    # Avoid division by zero for u2_prime (edge case where ray direction doesn't change)
    l2_prime = np.where(u2_prime != 0, (r2 * i2) / u2_prime + r2, np.inf)
    
    # Transition to third surface
    l3 = l2_prime - d2
    u3 = u2_prime
    
    # Ray tracing through third surface
    i3 = (l3 - r3) / r3 * u3 if r3 != 0 else 0
    i3_prime = (n2 / n_total) * i3
    u3_prime = u3 + i3 - i3_prime
    # Avoid division by zero for u3_prime
    l3_prime = np.where(u3_prime != 0, (r3 * i3) / u3_prime + r3, np.inf)
    
    # Axial image location relative to third lens (origin)
    l31 = l3_prime
    
    return l31
