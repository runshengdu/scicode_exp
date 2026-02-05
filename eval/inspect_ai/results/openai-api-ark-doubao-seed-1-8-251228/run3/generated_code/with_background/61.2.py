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
    x_pix, y_pix = p
    xc, yc = b_c
    
    # Calculate physical coordinates of the pixel in lab coordinate system (mm)
    X = det_d
    Y = - (x_pix - xc) * p_s
    Z = - (y_pix - yc) * p_s
    
    # Vector from sample to detector pixel
    r = np.array([X, Y, Z])
    r_mag = np.linalg.norm(r)
    
    # Magnitude of X-ray momentum (1/wavelength)
    k = 1.0 / wl
    
    # Compute scattered and incident momentum vectors
    k_s = (r / r_mag) * k
    k_i = np.array([k, 0.0, 0.0])
    
    # Calculate momentum transfer Q = k_s - k_i
    Q_vec = k_s - k_i
    
    # Reshape to 3x1 matrix
    Q = Q_vec.reshape(3, 1)
    
    return Q
