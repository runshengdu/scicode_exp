import numpy as np



def Bmat(pa):
    '''Calculate the B matrix.
    Input
    pa = (a,b,c,alpha,beta,gamma)
    a,b,c: the lengths a, b, and c of the three cell edges meeting at a vertex, float in the unit of angstrom
    alpha,beta,gamma: the angles alpha, beta, and gamma between those edges, float in the unit of degree
    Output
    B: a 3*3 matrix, float
    '''
    a, b, c, alpha, beta, gamma = pa
    
    # Convert angles from degrees to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)
    
    # Construct direct lattice vectors in Cartesian coordinates
    a1 = np.array([a, 0.0, 0.0])
    a2_x = b * np.cos(gamma_rad)
    a2_y = b * np.sin(gamma_rad)
    a2 = np.array([a2_x, a2_y, 0.0])
    
    a3_x = c * np.cos(beta_rad)
    numerator_y = np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)
    denominator_y = np.sin(gamma_rad)
    a3_y = c * numerator_y / denominator_y
    
    # Calculate z-component of a3, handle numerical precision
    a3_z_sq = c**2 - a3_x**2 - a3_y**2
    a3_z = np.sqrt(np.maximum(a3_z_sq, 0.0))
    a3 = np.array([a3_x, a3_y, a3_z])
    
    # Compute unit cell volume via scalar triple product
    V = np.dot(a1, np.cross(a2, a3))
    
    # Calculate reciprocal lattice vectors
    b1 = np.cross(a2, a3) / V
    b2 = np.cross(a3, a1) / V
    b3 = np.cross(a1, a2) / V
    
    # Form B matrix with reciprocal vectors as columns
    B = np.column_stack((b1, b2, b3))
    
    return B
