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
    
    cos_alpha = np.cos(alpha_rad)
    cos_beta = np.cos(beta_rad)
    cos_gamma = np.cos(gamma_rad)
    sin_gamma = np.sin(gamma_rad)
    
    # Compute unit cell volume using lattice parameters
    sqrt_term = 1.0 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma
    volume = a * b * c * np.sqrt(sqrt_term)
    
    # Construct direct lattice vectors in right-handed Cartesian coordinates
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([b * cos_gamma, b * sin_gamma, 0.0])
    
    x = c * cos_beta
    y = (c * (cos_alpha - cos_beta * cos_gamma)) / sin_gamma
    z = volume / (a * b * sin_gamma)
    a3 = np.array([x, y, z])
    
    # Calculate reciprocal lattice vectors
    b1 = np.cross(a2, a3) / volume
    b2 = np.cross(a3, a1) / volume
    b3 = np.cross(a1, a2) / volume
    
    # Construct B matrix with reciprocal vectors as columns
    B = np.column_stack((b1, b2, b3))
    
    return B
