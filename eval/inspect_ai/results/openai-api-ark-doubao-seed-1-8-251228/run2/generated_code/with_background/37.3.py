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
    h1 = np.asarray(h1)
    
    # Ray tracing through first surface (object at infinity, parallel rays)
    sin_I1 = h1 / r1
    sin_I1_prime = (n_total / n1) * sin_I1
    
    # Calculate incidence and refraction angles
    I1 = np.arcsin(sin_I1)
    I1_prime = np.arcsin(sin_I1_prime)
    
    # Calculate post-refraction aperture angle
    U1 = 0.0
    U1_prime = U1 + I1 - I1_prime
    
    # Calculate image distance after first surface, avoid division by zero
    sin_U1_prime = np.sin(U1_prime)
    L1_prime = np.where(sin_U1_prime != 0, r1 + r1 * (sin_I1_prime / sin_U1_prime), np.inf)
    
    # Transition to second surface
    L2 = L1_prime - d1
    U2 = U1_prime
    
    # Ray tracing through second surface
    sin_I2 = (L2 - r2) / r2 * np.sin(U2) if r2 != 0 else np.zeros_like(L2)
    sin_I2_prime = (n1 / n2) * sin_I2
    
    # Calculate incidence and refraction angles
    I2 = np.arcsin(sin_I2)
    I2_prime = np.arcsin(sin_I2_prime)
    
    # Calculate post-refraction aperture angle
    U2_prime = U2 + I2 - I2_prime
    
    # Calculate image distance after second surface, avoid division by zero
    sin_U2_prime = np.sin(U2_prime)
    L2_prime = np.where(sin_U2_prime != 0, r2 + r2 * (sin_I2_prime / sin_U2_prime), np.inf)
    
    # Transition to third surface
    L3 = L2_prime - d2
    U3 = U2_prime
    
    # Ray tracing through third surface
    sin_I3 = (L3 - r3) / r3 * np.sin(U3) if r3 != 0 else np.zeros_like(L3)
    sin_I3_prime = (n2 / n_total) * sin_I3
    
    # Calculate incidence and refraction angles
    I3 = np.arcsin(sin_I3)
    I3_prime = np.arcsin(sin_I3_prime)
    
    # Calculate post-refraction aperture angle
    U3_prime = U3 + I3 - I3_prime
    
    # Calculate image distance after third surface, avoid division by zero
    sin_U3_prime = np.sin(U3_prime)
    L3_prime = np.where(sin_U3_prime != 0, r3 + r3 * (sin_I3_prime / sin_U3_prime), np.inf)
    
    # Axial image location relative to third lens (origin)
    L31 = L3_prime
    
    return L31



def compute_LC(h1, r1, r2, r3, d1, d2, n1, n2, n3, n_total):
    '''Computes spherical aberration by comparing paraxial and axial calculations.
    Parameters:
    - h1 (array of floats): Aperture heights, in range (0.01, hm)
    - r1, r2, r3 (floats): Radii of curvature of the three surfaces
    - d1, d2 (floats): Separation distances between surfaces
    - n1, n2, n3 (floats): Refractive indices for the three lens materials
    - n_total (float): Refractive index of the surrounding medium (air)
    Returns:
    - LC (array of floats): Spherical aberration (difference between paraxial and axial)
    '''
    # Calculate paraxial axial positions
    paraxial_pos = calculate_paraxial(h1, r1, r2, r3, d1, d2, n1, n2, n3, n_total)
    # Calculate non-paraxial (axial) positions
    non_paraxial_pos = calculate_non_paraxial(h1, r1, r2, r3, d1, d2, n1, n2, n3, n_total)
    # Compute spherical aberration as the difference between paraxial and non-paraxial positions
    LC = paraxial_pos - non_paraxial_pos
    
    return LC
