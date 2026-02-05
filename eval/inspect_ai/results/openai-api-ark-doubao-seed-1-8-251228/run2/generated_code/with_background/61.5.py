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
    # Calculate B matrix using provided Bmat function
    B = Bmat(pa)
    
    # Compute q1 and q2 (reciprocal lattice vectors in Cartesian)
    H1_arr = np.array(H1).reshape(3, 1)
    q1 = B @ H1_arr
    H2_arr = np.array(H2).reshape(3, 1)
    q2 = B @ H2_arr
    
    # Compute t_c orthogonal unit-vector triple
    # t1c: unit vector along q1
    q1_norm = np.linalg.norm(q1)
    t1c = q1 / q1_norm
    
    # t3c: unit vector along q1 × q2
    cross_q1q2 = np.cross(q1.flatten(), q2.flatten()).reshape(3, 1)
    cross_q1q2_norm = np.linalg.norm(cross_q1q2)
    t3c = cross_q1q2 / cross_q1q2_norm
    
    # t2c: unit vector along t3c × t1c to form orthogonal triple
    t2c = np.cross(t3c.flatten(), t1c.flatten()).reshape(3, 1)
    t2c_norm = np.linalg.norm(t2c)
    if t2c_norm > 1e-10:
        t2c = t2c / t2c_norm
    t_c = (t1c, t2c, t3c)
    
    # Compute measured momentum transfers Q_meas1 and Q_meas2 using q_cal
    Q_meas1 = q_cal(p1, b_c, det_d, p_s, wl)
    Q_meas2 = q_cal(p2, b_c, det_d, p_s, wl)
    
    # Calculate rotation angles in radians
    theta1_deg = z1 * z_s
    theta2_deg = z2 * z_s
    theta1_rad = np.deg2rad(theta1_deg)
    theta2_rad = np.deg2rad(theta2_deg)
    
    # Construct rotation matrices for rotation around -y axis
    # R_y(-theta) for theta1
    cos1, sin1 = np.cos(theta1_rad), np.sin(theta1_rad)
    R1 = np.array([
        [cos1, 0, -sin1],
        [0, 1, 0],
        [sin1, 0, cos1]
    ])
    
    # R_y(-theta) for theta2
    cos2, sin2 = np.cos(theta2_rad), np.sin(theta2_rad)
    R2 = np.array([
        [cos2, 0, -sin2],
        [0, 1, 0],
        [sin2, 0, cos2]
    ])
    
    # Compute Q0 (momentum transfer before rotation) by rotating back
    Q0_1 = R1.T @ Q_meas1
    Q0_2 = R2.T @ Q_meas2
    
    # Compute t_g orthogonal unit-vector triple
    # t1g: unit vector along Q0_1
    Q0_1_norm = np.linalg.norm(Q0_1)
    t1g = Q0_1 / Q0_1_norm
    
    # t3g: unit vector along Q0_1 × Q0_2
    cross_Q01Q02 = np.cross(Q0_1.flatten(), Q0_2.flatten()).reshape(3, 1)
    cross_Q01Q02_norm = np.linalg.norm(cross_Q01Q02)
    t3g = cross_Q01Q02 / cross_Q01Q02_norm
    
    # t2g: unit vector along t3g × t1g to form orthogonal triple
    t2g = np.cross(t3g.flatten(), t1g.flatten()).reshape(3, 1)
    t2g_norm = np.linalg.norm(t2g)
    if t2g_norm > 1e-10:
        t2g = t2g / t2g_norm
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
    # Construct matrices T_c and T_g with basis vectors as columns
    T_c = np.hstack(t_c)
    T_g = np.hstack(t_g)
    
    # Calculate U using the relation T_g = U @ T_c, leveraging orthogonality of T_c (T_c.T = T_c^{-1})
    U = T_g @ T_c.T
    
    return U



def get_hkl(p, z, b_c, det_d, p_s, wl, pa, H1, H2, p1, p2, z1, z2, z_s):
    '''Convert pixel (x,y) at frame z to reciprocal space (h,k,l)
    Input
    The Bragg peak to be indexed:
    p: detector pixel (x,y), a tuple of two integer
    z: frame number, integer
    instrument configuration:
    b_c: incident beam center at detector pixel (xc,yc), a tuple of float
    det_d: sample distance to the detector, float in the unit of mm
    p_s: detector pixel size, and each pixel is a square, float in the unit of mm
    wl: X-ray wavelength, float in the unit of angstrom
    crystal structure:
    pa = (a,b,c,alpha,beta,gamma)
    a,b,c: the lengths a, b, and c of the three cell edges meeting at a vertex, float in the unit of angstrom
    alpha,beta,gamma: the angles alpha, beta, and gamma between those edges, float in the unit of degree
    The two Bragg peaks used for orienting the crystal:
    H1 = (h1,k1,l1),primary reflection, h1,k1,l1 is integer
    H2 = (h2,k2,l2),secondary reflection, h2,k2,l2 is integer
    p1: detector pixel (x1,y1), a tuple of two integer
    p2: detector pixel (x2,y2), a tuple of two integer
    z1,z2: frame number, integer
    z_s: step size in the       heta rotation, float in the unit of degree
    Output
    q: 3x1 orthogonal matrix, float
    '''
    # Calculate B matrix using provided Bmat function
    B = Bmat(pa)
    
    # Calculate orthogonal basis triples using u_triple function
    t_c, t_g = u_triple(pa, H1, H2, p1, p2, b_c, det_d, p_s, wl, z1, z2, z_s)
    
    # Calculate orientation matrix U using Umat function
    U = Umat(t_c, t_g)
    
    # Calculate measured momentum transfer Q using q_cal function
    Q = q_cal(p, b_c, det_d, p_s, wl)
    
    # Calculate rotation angle for current frame (convert to radians)
    theta_deg = z * z_s
    theta_rad = np.deg2rad(theta_deg)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    
    # Construct rotation matrix G (rotation around -y axis by theta)
    G = np.array([
        [cos_theta, 0.0, -sin_theta],
        [0.0, 1.0, 0.0],
        [sin_theta, 0.0, cos_theta]
    ])
    
    # Rotate Q back to unrotated orientation (G inverse is transpose for orthogonal matrices)
    Q_rotated_back = G.T @ Q
    
    # Compute UB matrix and its inverse
    UB = U @ B
    UB_inv = np.linalg.inv(UB)
    
    # Calculate (h,k,l) coordinates
    hkl = UB_inv @ Q_rotated_back
    
    return hkl
