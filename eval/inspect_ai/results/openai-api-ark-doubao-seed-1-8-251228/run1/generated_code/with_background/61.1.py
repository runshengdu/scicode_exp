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
    
    # Compute direct lattice vectors in Cartesian coordinates
    a1 = np.array([a, 0.0, 0.0])
    
    a2_x = b * np.cos(gamma_rad)
    a2_y = b * np.sin(gamma_rad)
    a2 = np.array([a2_x, a2_y, 0.0])
    
    a3_x = c * np.cos(beta_rad)
    # Solve for a3_y using the dot product condition a2 · a3 = b*c*cos(alpha)
    numerator_a3y = b * c * np.cos(alpha_rad) - a2_x * a3_x
    a3_y = numerator_a3y / a2_y
    # Compute a3_z using the length of a3 vector
    a3_z_sq = c**2 - a3_x**2 - a3_y**2
    a3_z = np.sqrt(a3_z_sq)
    a3 = np.array([a3_x, a3_y, a3_z])
    
    # Calculate unit cell volume via scalar triple product
    V = np.dot(a1, np.cross(a2, a3))
    
    # Compute reciprocal lattice vectors using the given convention
    b1 = np.cross(a2, a3) / V
    b2 = np.cross(a3, a1) / V
    b3 = np.cross(a1, a2) / V
    
    # Construct B matrix by stacking reciprocal vectors as columns
    B = np.column_stack((b1, b2, b3))
    
    return B
