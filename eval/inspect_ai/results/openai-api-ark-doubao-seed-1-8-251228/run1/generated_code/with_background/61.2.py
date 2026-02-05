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



def q_cal(p, b_c, det_d, p_s, wl):
    '''Calculate the momentum transfer Q at detector pixel (x,y). Here we're employing the convention, k=1/\lambda,
    k represents the x-ray momentum and \lambda denotes the wavelength.
    Input
    p: detector pixel (x,y), a tuple of two integer
    b_c: incident beam center at detector pixel (xc,yc), a tuple of float
    det_d: sample distance to the detector, float in the unit of mm
    p_s: detector pixel size, and each pixel is a square, float in the unit of mm
    wl: X-ray wavelength, float in the unit of angstrom
    Output
    Q: a 3x1 matrix, float in the unit of inverse angstrom
    '''
    x_p, y_p = p
    xc, yc = b_c
    
    # Calculate physical offset from beam center in detector coordinates (mm)
    delta_x_det = (x_p - xc) * p_s
    delta_y_det = (y_p - yc) * p_s
    
    # Convert detector coordinate offsets to lab coordinate system (mm)
    y_lab = -delta_x_det
    z_lab = -delta_y_det
    x_lab = det_d
    
    # Position vector from sample to detector pixel in lab coordinates
    r = np.array([x_lab, y_lab, z_lab])
    r_mag = np.linalg.norm(r)
    r_hat = r / r_mag  # Unit vector in scattered beam direction
    
    # Calculate X-ray momentum magnitude (1/Å)
    k = 1.0 / wl
    
    # Momentum vectors for scattered and incident beams
    k_s = k * r_hat
    k_i = np.array([k, 0.0, 0.0])
    
    # Compute momentum transfer Q = k_s - k_i
    Q_vec = k_s - k_i
    
    # Reshape to 3x1 matrix
    Q = Q_vec.reshape((3, 1))
    
    return Q
