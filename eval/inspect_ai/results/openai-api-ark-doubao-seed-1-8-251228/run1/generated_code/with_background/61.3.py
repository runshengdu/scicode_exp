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
    # Compute B matrix using provided Bmat function
    B = Bmat(pa)
    
    # Calculate reciprocal lattice vectors q1 and q2 in Cartesian coordinates
    H1_col = np.array(H1).reshape((3, 1))
    q1 = B @ H1_col
    H2_col = np.array(H2).reshape((3, 1))
    q2 = B @ H2_col
    
    # Construct orthogonal unit-vector triple t_c
    # t1c: unit vector along q1
    q1_norm = np.linalg.norm(q1)
    t1c = q1 / q1_norm
    
    # t3c: unit vector along q1 × q2
    q1_cross_q2 = np.cross(q1.flatten(), q2.flatten()).reshape((3, 1))
    q1q2_cross_norm = np.linalg.norm(q1_cross_q2)
    t3c = q1_cross_q2 / q1q2_cross_norm
    
    # t2c: unit vector orthogonal to t1c and t3c (right-handed system)
    t2c = np.cross(t3c.flatten(), t1c.flatten()).reshape((3, 1))
    
    # Construct orthogonal unit-vector triple t_g
    # Calculate rotation angles for frames z1 and z2
    theta1 = z1 * z_s
    theta2 = z2 * z_s
    theta1_rad = np.radians(theta1)
    theta2_rad = np.radians(theta2)
    
    # Define y-axis rotation matrices for rotating back to original orientation
    c1, s1 = np.cos(theta1_rad), np.sin(theta1_rad)
    R1 = np.array([
        [c1, 0, s1],
        [0, 1, 0],
        [-s1, 0, c1]
    ])
    
    c2, s2 = np.cos(theta2_rad), np.sin(theta2_rad)
    R2 = np.array([
        [c2, 0, s2],
        [0, 1, 0],
        [-s2, 0, c2]
    ])
    
    # Calculate observed momentum transfers from detector positions
    Q1_obs = q_cal(p1, b_c, det_d, p_s, wl)
    Q2_obs = q_cal(p2, b_c, det_d, p_s, wl)
    
    # Rotate observed Q back to original crystal orientation
    Q1 = R1 @ Q1_obs
    Q2 = R2 @ Q2_obs
    
    # t1g: unit vector along Q1
    Q1_norm = np.linalg.norm(Q1)
    t1g = Q1 / Q1_norm
    
    # t3g: unit vector along Q1 × Q2
    Q1_cross_Q2 = np.cross(Q1.flatten(), Q2.flatten()).reshape((3, 1))
    Q1Q2_cross_norm = np.linalg.norm(Q1_cross_Q2)
    t3g = Q1_cross_Q2 / Q1Q2_cross_norm
    
    # t2g: unit vector orthogonal to t1g and t3g (right-handed system)
    t2g = np.cross(t3g.flatten(), t1g.flatten()).reshape((3, 1))
    
    # Package results into required format
    t_c = (t1c, t2c, t3c)
    t_g = (t1g, t2g, t3g)
    
    return (t_c, t_g)
