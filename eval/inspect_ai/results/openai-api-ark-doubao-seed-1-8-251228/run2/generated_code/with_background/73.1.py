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
    a, b, c, alpha_deg, beta_deg, gamma_deg = pa
    # Convert angles from degrees to radians
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)
    
    # Construct direct lattice vectors in Cartesian coordinates
    a1 = np.array([a, 0.0, 0.0])
    a2_x = b * np.cos(gamma)
    a2_y = b * np.sin(gamma)
    a2 = np.array([a2_x, a2_y, 0.0])
    
    a3_x = c * np.cos(beta)
    numerator_a3y = np.cos(alpha) - np.cos(beta) * np.cos(gamma)
    denominator_a3y = np.sin(gamma)
    a3_y = c * numerator_a3y / denominator_a3y
    a3_z_sq = c**2 - a3_x**2 - a3_y**2
    a3_z = np.sqrt(a3_z_sq)
    a3 = np.array([a3_x, a3_y, a3_z])
    
    # Calculate unit cell volume
    V = np.dot(a1, np.cross(a2, a3))
    
    # Compute reciprocal lattice vectors
    b1 = np.cross(a2, a3) / V
    b2 = np.cross(a3, a1) / V
    b3 = np.cross(a1, a2) / V
    
    # Construct B matrix with reciprocal vectors as columns
    B = np.column_stack((b1, b2, b3))
    
    return B
