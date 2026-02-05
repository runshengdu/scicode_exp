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



def u_triple(pa, H1, H2, p1, p2, b_c, det_d, p_s, wl, z1, z2, z_s):
    '''Calculate two orthogonal unit-vector triple t_i_c and t_i_g. Frame z starts from 0
    Input
    pa = (a,b,c,alpha,beta,gamma)
    a,b,c: the lengths a, b, and c of the three cell edges meeting at a vertex, float in the unit of angstrom
    alpha,beta,gamma: the angles alpha, beta, and gamma between those edges, float in the unit of degree
    H1 = (h1,k1,l1),primary reflection, h1,k1,l1 is integer
    H2 = (h2,k2,l2),secondary reflection, h2,k2,l2 is integer
    p1: detector pixel (x1,y1), a tuple of two integer
    p2: detector pixel (x2,y2), a tuple of two integer
    b_c: incident beam center at detector pixel (xc,yc), a tuple of float
    det_d: sample distance to the detector, float in the unit of mm
    p_s: detector pixel size, and each pixel is a square, float in the unit of mm
    wl: X-ray wavelength, float in the unit of angstrom
    z1,z2: frame number, integer
    z_s: step size in the \phi rotation, float in the unit of degree
    Output
    t_c_t_g: tuple (t_c,t_g), t_c = (t1c,t2c,t3c) and t_g = (t1g,t2g,t3g).
    Each element inside t_c and t_g is a 3x1 matrix, float
    '''
    # Calculate B matrix using provided function
    B = Bmat(pa)
    
    # Compute t_c triple (crystal-local Cartesian system)
    # Get reciprocal lattice vectors in crystal Cartesian coordinates
    q1 = B @ np.array(H1)
    q2 = B @ np.array(H2)
    
    # Unit vector along q1
    q1_norm = np.linalg.norm(q1)
    t1c_1d = q1 / q1_norm
    
    # Unit vector along q1 × q2
    q1_cross_q2 = np.cross(q1, q2)
    q1_cross_q2_norm = np.linalg.norm(q1_cross_q2)
    t3c_1d = q1_cross_q2 / q1_cross_q2_norm
    
    # Unit vector orthogonal to t1c and t3c (right-handed triple)
    t2c_1d = np.cross(t3c_1d, t1c_1d)
    
    # Reshape to 3x1 matrices
    t1c = t1c_1d.reshape(3, 1)
    t2c = t2c_1d.reshape(3, 1)
    t3c = t3c_1d.reshape(3, 1)
    t_c = (t1c, t2c, t3c)
    
    # Compute t_g triple (lab coordinate system before rotation)
    # Get measured momentum transfers from detector
    Q1_measured = q_cal(p1, b_c, det_d, p_s, wl)
    Q1_measured_1d = Q1_measured.flatten()
    
    Q2_measured = q_cal(p2, b_c, det_d, p_s, wl)
    Q2_measured_1d = Q2_measured.flatten()
    
    # Rotation matrix for frame z1 (rotation around -y by z1*z_s degrees)
    theta_z1 = z1 * z_s
    theta_z1_rad = np.radians(theta_z1)
    cos_theta1 = np.cos(theta_z1_rad)
    sin_theta1 = np.sin(theta_z1_rad)
    R_z1 = np.array([
        [cos_theta1, 0, -sin_theta1],
        [0, 1, 0],
        [sin_theta1, 0, cos_theta1]
    ])
    
    # Rotation matrix for frame z2
    theta_z2 = z2 * z_s
    theta_z2_rad = np.radians(theta_z2)
    cos_theta2 = np.cos(theta_z2_rad)
    sin_theta2 = np.sin(theta_z2_rad)
    R_z2 = np.array([
        [cos_theta2, 0, -sin_theta2],
        [0, 1, 0],
        [sin_theta2, 0, cos_theta2]
    ])
    
    # Transform measured Q back to initial (unrotated) lab coordinates
    Q1_1d = R_z1.T @ Q1_measured_1d
    Q2_1d = R_z2.T @ Q2_measured_1d
    
    # Unit vector along Q1
    Q1_norm = np.linalg.norm(Q1_1d)
    t1g_1d = Q1_1d / Q1_norm
    
    # Unit vector along Q1 × Q2
    Q1_cross_Q2 = np.cross(Q1_1d, Q2_1d)
    Q1_cross_Q2_norm = np.linalg.norm(Q1_cross_Q2)
    t3g_1d = Q1_cross_Q2 / Q1_cross_Q2_norm
    
    # Unit vector orthogonal to t1g and t3g (right-handed triple)
    t2g_1d = np.cross(t3g_1d, t1g_1d)
    
    # Reshape to 3x1 matrices
    t1g = t1g_1d.reshape(3, 1)
    t2g = t2g_1d.reshape(3, 1)
    t3g = t3g_1d.reshape(3, 1)
    t_g = (t1g, t2g, t3g)
    
    return (t_c, t_g)



def Umat(t_c, t_g):
    '''Write down the orientation matrix which transforms from bases t_c to t_g
    Input
    t_c, tuple with three elements, each element is a 3x1 matrix, float
    t_g, tuple with three elements, each element is a 3x1 matrix, float
    Output
    U: 3x3 orthogonal matrix, float
    '''
    # Construct matrix T_c with columns as t_c vectors
    T_c = np.hstack(t_c)
    # Construct matrix T_g with columns as t_g vectors
    T_g = np.hstack(t_g)
    # Calculate orientation matrix U using the relation T_g = U @ T_c
    # Since T_c is orthogonal, its inverse is its transpose
    U = T_g @ T_c.T
    return U
