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
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    gamma_rad = np.deg2rad(gamma)
    
    # Compute direct lattice vectors in Cartesian coordinates
    a1 = np.array([a, 0.0, 0.0])
    a2_x = b * np.cos(gamma_rad)
    a2_y = b * np.sin(gamma_rad)
    a2 = np.array([a2_x, a2_y, 0.0])
    
    a3_x = c * np.cos(beta_rad)
    numerator_a3y = np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)
    denominator_a3y = np.sin(gamma_rad)
    a3_y = c * numerator_a3y / denominator_a3y
    
    # Calculate a3_z, ensuring non-negative value under square root
    a3_z_sq = c**2 - a3_x**2 - a3_y**2
    a3_z_sq = np.maximum(a3_z_sq, 0.0)
    a3_z = np.sqrt(a3_z_sq)
    a3 = np.array([a3_x, a3_y, a3_z])
    
    # Compute unit cell volume
    a2_cross_a3 = np.cross(a2, a3)
    V = np.dot(a1, a2_cross_a3)
    
    # Calculate reciprocal lattice vectors
    b1 = a2_cross_a3 / V
    b2 = np.cross(a3, a1) / V
    b3 = np.cross(a1, a2) / V
    
    # Construct B matrix with reciprocal vectors as columns
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
    Q: a 3*3 matrix, float
    '''
    x, y = p
    xc, yc = b_c
    
    # Calculate pixel offsets from beam center
    dx_pixel = x - xc
    dy_pixel = y - yc
    
    # Compute position vector components in lab coordinates (mm)
    r_x = det_d
    r_y = -dx_pixel * p_s
    r_z = -dy_pixel * p_s
    r = np.array([r_x, r_y, r_z])
    
    # Calculate magnitude of the position vector
    r_mag = np.linalg.norm(r)
    
    # Compute unit vector in the direction of the scattered beam
    u = r / r_mag
    
    # Calculate momentum vectors (k = 1/λ)
    k_mag = 1.0 / wl
    k_s = k_mag * u
    k_i = np.array([k_mag, 0.0, 0.0])
    
    # Calculate momentum transfer Q = k_s - k_i
    Q = k_s - k_i
    
    # Reshape to 3x1 matrix
    Q = Q.reshape(3, 1)
    
    return Q
