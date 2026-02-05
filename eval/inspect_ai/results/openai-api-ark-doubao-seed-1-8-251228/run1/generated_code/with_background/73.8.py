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
    
    # Compute the term for unit cell volume calculation
    S = 1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma
    sqrt_S = np.sqrt(S)
    
    # Construct direct lattice vectors in Cartesian coordinates
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([b * cos_gamma, b * sin_gamma, 0.0])
    a3x = c * cos_beta
    a3y = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    a3z = c * sqrt_S / sin_gamma
    a3 = np.array([a3x, a3y, a3z])
    
    # Calculate unit cell volume
    V = np.dot(a1, np.cross(a2, a3))
    
    # Compute reciprocal lattice vectors
    b1 = np.cross(a2, a3) / V
    b2 = np.cross(a3, a1) / V
    b3 = np.cross(a1, a2) / V
    
    # Construct B matrix with reciprocal vectors as columns
    B = np.column_stack((b1, b2, b3))
    
    return B


def q_cal_p(p, b_c, det_d, p_s, wl, yaw, pitch, roll):
    '''Calculate the momentum transfer Q at detector pixel (x,y). Here we use the convention of k=1/\lambda,
    k and \lambda are the x-ray momentum and wavelength respectively
    Input
    p: detector pixel (x,y), a tuple of two integer
    b_c: incident beam center at detector pixel (xc,yc), a tuple of float
    det_d: sample distance to the detector, float in the unit of mm
    p_s: detector pixel size, and each pixel is a square, float in the unit of mm
    wl: X-ray wavelength, float in the unit of angstrom
    yaw,pitch,roll: rotation angles of the detector, float in the unit of degree
    Output
    Q: a 3x1 matrix, float in the unit of inverse angstrom
    '''
    x, y = p
    xc, yc = b_c
    
    # Calculate detector pixel position in detector coordinate system (DCS)
    x_det = (x - xc) * p_s
    y_det = (y - yc) * p_s
    z_det = det_d
    P_DCS = np.array([[x_det], [y_det], [z_det]])
    
    # Default transformation matrix from DCS to lab coordinate system (LCS)
    M = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ])
    
    # Convert rotation angles to radians
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)
    
    # Compute cosine and sine of rotation angles
    cos_roll = np.cos(roll_rad)
    sin_roll = np.sin(roll_rad)
    cos_pitch = np.cos(pitch_rad)
    sin_pitch = np.sin(pitch_rad)
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    
    # Build rotation matrices for roll (x-axis), pitch (y-axis), yaw (z-axis)
    R_x = np.array([
        [1, 0, 0],
        [0, cos_roll, -sin_roll],
        [0, sin_roll, cos_roll]
    ])
    
    R_y = np.array([
        [cos_pitch, 0, sin_pitch],
        [0, 1, 0],
        [-sin_pitch, 0, cos_pitch]
    ])
    
    R_z = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])
    
    # Compute detector rotation matrix D (extrinsic rotations: yaw -> pitch -> roll)
    D = R_x @ R_y @ R_z
    
    # Transform pixel position from DCS to LCS
    P_LCS = D @ M @ P_DCS
    
    # Compute unit vector in the direction from sample to detector pixel
    dist = np.linalg.norm(P_LCS)
    u = P_LCS / dist
    
    # Compute wave vector magnitude (k = 1/λ)
    k = 1.0 / wl
    
    # Compute k_s (scattered wave vector) and k_i (incident wave vector)
    k_s = k * u
    k_i = np.array([[k], [0.0], [0.0]])
    
    # Compute momentum transfer Q = k_s - k_i
    Q = k_s - k_i
    
    return Q



def Umat(t_c, t_g):
    '''Write down the orientation matrix which transforms from bases t_c to t_g
    Input
    t_c, tuple with three elements, each element is a 3x1 matrix, float
    t_g, tuple with three elements, each element is a 3x1 matrix, float
    Output
    U: 3x3 orthogonal matrix, float
    '''
    # Construct transformation matrices with basis vectors as columns
    T_c = np.column_stack(t_c)
    T_g = np.column_stack(t_g)
    
    # Calculate U using the relation T_g = U @ T_c, leveraging orthogonality of T_c (inverse = transpose)
    U = T_g @ T_c.T
    
    return U



def get_hkl_p(p, z, b_c, det_d, p_s, wl, yaw, pitch, roll, pa, H1, H2, p1, p2, z1, z2, z_s, chi, phi):
    '''Convert pixel (x,y) at frame z to reciprocal space (h,k,l)
    Input
    p: detector pixel (x,y), a tuple of two integer
    z: frame number, integer
    b_c: incident beam center at detector pixel (xc,yc), a tuple of float
    det_d: sample distance to the detector, float in the unit of mm
    p_s: detector pixel size, and each pixel is a square, float in the unit of mm
    wl: X-ray wavelength, float in the unit of angstrom
    yaw,pitch,roll: rotation angles of the detector, float in the unit of degree
    pa = (a,b,c,alpha,beta,gamma)
    a,b,c: the lengths a, b, and c of the three cell edges meeting at a vertex, float in the unit of angstrom
    alpha,beta,gamma: the angles alpha, beta, and gamma between those edges, float in the unit of degree
    H1 = (h1,k1,l1),primary reflection, h1,k1,l1 is integer
    H2 = (h2,k2,l2),secondary reflection, h2,k2,l2 is integer
    p1: detector pixel (x1,y1), a tuple of two integer
    p2: detector pixel (x2,y2), a tuple of two integer
    z1,z2: frame number, integer
    z_s: step size in the \phi rotation, float in the unit of degree
    chi,phi: diffractometer angles, float in the unit of degree
    Output
    q: 3x1 matrix, float
    '''
    # Step 1: Compute B matrix from lattice parameters
    B = Bmat(pa)
    
    # Step 2: Compute reciprocal lattice vectors q1, q2 for known reflections
    h1, k1, l1 = H1
    h2, k2, l2 = H2
    q1 = B @ np.array([[h1], [k1], [l1]])
    q2 = B @ np.array([[h2], [k2], [l2]])
    
    # Step 3: Compute momentum transfer Q1, Q2 for known detector positions
    Q1 = q_cal_p(p1, b_c, det_d, p_s, wl, yaw, pitch, roll)
    Q2 = q_cal_p(p2, b_c, det_d, p_s, wl, yaw, pitch, roll)
    
    # Step 4: Compute delta_theta and rotate Q2 by delta_theta around y-axis
    delta_theta_deg = (z2 - z1) * z_s
    delta_theta_rad = np.radians(delta_theta_deg)
    cos_dt, sin_dt = np.cos(delta_theta_rad), np.sin(delta_theta_rad)
    R_y_dt = np.array([
        [cos_dt, 0, sin_dt],
        [0, 1, 0],
        [-sin_dt, 0, cos_dt]
    ])
    Q2_rot = R_y_dt @ Q2
    
    # Step 5: Construct T_c basis from q1, q2
    # t1^c: unit vector along q1
    q1_norm = np.linalg.norm(q1)
    t1c = q1 / q1_norm
    # t3^c: unit vector along q1 × q2
    cross_q1q2 = np.cross(q1.flatten(), q2.flatten())[:, np.newaxis]
    cross_q1q2_norm = np.linalg.norm(cross_q1q2)
    t3c = cross_q1q2 / cross_q1q2_norm
    # t2^c: unit vector along t3^c × t1^c
    t2c = np.cross(t3c.flatten(), t1c.flatten())[:, np.newaxis]
    # Assemble T_c matrix
    T_c = np.column_stack((t1c, t2c, t3c))
    
    # Step 6: Construct T_g' basis from Q1, Q2_rot
    # t1^g': unit vector along Q1
    Q1_norm = np.linalg.norm(Q1)
    t1g = Q1 / Q1_norm
    # t3^g': unit vector along Q1 × Q2_rot
    cross_Q1Q2rot = np.cross(Q1.flatten(), Q2_rot.flatten())[:, np.newaxis]
    cross_Q1Q2rot_norm = np.linalg.norm(cross_Q1Q2rot)
    t3g = cross_Q1Q2rot / cross_Q1Q2rot_norm
    # t2^g': unit vector along t3^g' × t1^g'
    t2g = np.cross(t3g.flatten(), t1g.flatten())[:, np.newaxis]
    # Assemble T_g' matrix
    T_g_prime = np.column_stack((t1g, t2g, t3g))
    
    # Step 7: Compute rotation matrix M
    M = T_g_prime @ T_c.T
    
    # Step 8: Process target pixel and frame
    # Compute Q for target pixel
    Q = q_cal_p(p, b_c, det_d, p_s, wl, yaw, pitch, roll)
    # Compute delta_theta for target frame and rotate Q
    delta_theta_z_deg = (z - z1) * z_s
    delta_theta_z_rad = np.radians(delta_theta_z_deg)
    cos_dt_z, sin_dt_z = np.cos(delta_theta_z_rad), np.sin(delta_theta_z_rad)
    R_y_dt_z = np.array([
        [cos_dt_z, 0, sin_dt_z],
        [0, 1, 0],
        [-sin_dt_z, 0, cos_dt_z]
    ])
    Q_rot = R_y_dt_z @ Q
    
    # Step 9: Compute (h,k,l) using the derived formula
    # Invert M (using transpose since M is orthogonal)
    M_inv = M.T
    temp = M_inv @ Q_rot
    # Invert B matrix
    B_inv = np.linalg.inv(B)
    # Compute final (h,k,l)
    q_tilde = B_inv @ temp
    
    return q_tilde

# Required helper functions from previous steps (assumed to be implemented)
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
    
    # Compute the term for unit cell volume calculation
    S = 1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma
    sqrt_S = np.sqrt(S)
    
    # Construct direct lattice vectors in Cartesian coordinates
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([b * cos_gamma, b * sin_gamma, 0.0])
    a3x = c * cos_beta
    a3y = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    a3z = c * sqrt_S / sin_gamma
    a3 = np.array([a3x, a3y, a3z])
    
    # Calculate unit cell volume
    V = np.dot(a1, np.cross(a2, a3))
    
    # Compute reciprocal lattice vectors
    b1 = np.cross(a2, a3) / V
    b2 = np.cross(a3, a1) / V
    b3 = np.cross(a1, a2) / V
    
    # Construct B matrix with reciprocal vectors as columns
    B = np.column_stack((b1, b2, b3))
    
    return B

def q_cal_p(p, b_c, det_d, p_s, wl, yaw, pitch, roll):
    '''Calculate the momentum transfer Q at detector pixel (x_det,y_det) in the lab coordinate system.
    Input
    p: detector pixel (x,y), a tuple of two integer
    b_c: incident beam center at detector pixel (xc,yc), a tuple of float
    det_d: sample distance to the detector, float in the unit of mm
    p_s: detector pixel size, and each pixel is a square, float in the unit of mm
    wl: X-ray wavelength, float in the unit of angstrom
    yaw,pitch,roll: rotation angles of the detector, float in the unit of degree
    Output
    Q: a 3x1 matrix, float in the unit of inverse angstrom
    '''

    x, y = p
    xc, yc = b_c
    
    # Calculate detector pixel position in detector coordinate system (DCS)
    x_det = (x - xc) * p_s
    y_det = (y - yc) * p_s
    z_det = det_d
    P_DCS = np.array([[x_det], [y_det], [z_det]])
    
    # Default transformation matrix from DCS to lab coordinate system (LCS)
    M = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ])
    
    # Convert rotation angles to radians
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)
    
    # Compute cosine and sine of rotation angles
    cos_roll = np.cos(roll_rad)
    sin_roll = np.sin(roll_rad)
    cos_pitch = np.cos(pitch_rad)
    sin_pitch = np.sin(pitch_rad)
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    
    # Build rotation matrices for roll (x-axis), pitch (y-axis), yaw (z-axis)
    R_x = np.array([
        [1, 0, 0],
        [0, cos_roll, -sin_roll],
        [0, sin_roll, cos_roll]
    ])
    
    R_y = np.array([
        [cos_pitch, 0, sin_pitch],
        [0, 1, 0],
        [-sin_pitch, 0, cos_pitch]
    ])
    
    R_z = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])
    
    # Compute detector rotation matrix D (extrinsic rotations: yaw -> pitch -> roll)
    D = R_x @ R_y @ R_z
    
    # Transform pixel position from DCS to LCS
    P_LCS = D @ M @ P_DCS
    
    # Compute unit vector in the direction from sample to detector pixel
    dist = np.linalg.norm(P_LCS)
    u = P_LCS / dist
    
    # Compute wave vector magnitude (k = 1/λ)
    k = 1.0 / wl
    
    # Compute k_s (scattered wave vector) and k_i (incident wave vector)
    k_s = k * u
    k_i = np.array([[k], [0.0], [0.0]])
    
    # Compute momentum transfer Q = k_s - k_i
    Q = k_s - k_i
    
    return Q


def ringdstar(pa, polar_max, wl):
    '''List all d*<d*_max and the corresponding (h,k,l). d*_max is determined by the maximum scattering angle
    and the x-ray wavelength
    Input
    pa = (a,b,c,alpha,beta,gamma)
    a,b,c: the lengths a, b, and c of the three cell edges meeting at a vertex, float in the unit of angstrom
    alpha,beta,gamma: the angles alpha, beta, and gamma between those edges, float in the unit of degree
    polar_max: maximum scattering angle, i.e. maximum angle between the x-ray beam axis
               and the powder ring, float in the unit of degree
    wl: X-ray wavelength, float in the unit of angstrom
    Output
    ringhkls: a dictionary, key is d* and each item is a sorted list with element of corresponding (h,k,l)
    '''
    # Compute B matrix using provided Bmat function
    B = Bmat(pa)
    
    # Calculate d*_max from polar_max and wavelength
    polar_max_rad = np.radians(polar_max)
    dstar_max = 2 * np.sin(polar_max_rad / 2) / wl
    
    # Compute Gram matrix of reciprocal lattice vectors
    G = B.T @ B
    
    # Find minimum eigenvalue of Gram matrix to determine safe upper bound for h,k,l
    eigenvalues = np.linalg.eigvalsh(G)
    lambda_min = np.min(eigenvalues)
    
    # Determine maximum range for h, k, l
    if lambda_min < 1e-12:
        M = 0
    else:
        M_squared = (dstar_max ** 2) / lambda_min
        M = int(np.floor(np.sqrt(M_squared)))
    
    ringhkls = {}
    
    # Iterate over all possible (h,k,l) triplets
    for h in range(-M, M + 1):
        for k in range(-M, M + 1):
            for l in range(-M, M + 1):
                # Skip the (0,0,0) triplet
                if h == 0 and k == 0 and l == 0:
                    continue
                
                # Calculate squared magnitude of reciprocal lattice vector
                hkl = np.array([h, k, l])
                q_squared = np.dot(hkl, G @ hkl)
                
                # Check if d* < d*_max (using squared values to avoid sqrt and reduce precision issues)
                if q_squared < (dstar_max ** 2) - 1e-12:
                    dstar = np.sqrt(q_squared)
                    # Round to 9 decimal places to group similar d* values
                    key = round(dstar, 9)
                    hkl_tuple = (h, k, l)
                    
                    if key not in ringhkls:
                        ringhkls[key] = []
                    ringhkls[key].append(hkl_tuple)
    
    # Sort each list of (h,k,l) triplets lexicographically
    for key in ringhkls:
        ringhkls[key].sort()
    
    return ringhkls


def hkl_pairs(pa, p1, p2, b_c, det_d, p_s, wl, yaw, pitch, roll, polar_max):
    '''Find the possible (h,k,l) for a pair of Bragg reflections (Q1,Q2)
    Input
    pa = (a,b,c,alpha,beta,gamma)
    a,b,c: the lengths a, b, and c of the three cell edges meeting at a vertex, float in the unit of angstrom
    alpha,beta,gamma: the angles alpha, beta, and gamma between those edges, float in the unit of degree
    p1: detector pixel (x1,y1), a tuple of two integer
    p2: detector pixel (x2,y2), a tuple of two integer
    b_c: incident beam center at detector pixel (xc,yc), a tuple of float
    det_d: sample distance to the detector, float in the unit of mm
    p_s: detector pixel size, and each pixel is a square, float in the unit of mm
    wl: X-ray wavelength, float in the unit of angstrom
    yaw,pitch,roll: rotation angles of the detector, float in the unit of degree
    polar_max: maximum scattering angle, i.e. maximum angle between the x-ray beam axis
               and the powder ring, float in the unit of degree
    Output
    (ha,hb): tuple (ha,hb). ha,hb is a list of possible sorted (h,k,l)
    '''
    # Step 1: Compute Q1 and Q2 for the given detector pixels
    Q1 = q_cal_p(p1, b_c, det_d, p_s, wl, yaw, pitch, roll)
    Q2 = q_cal_p(p2, b_c, det_d, p_s, wl, yaw, pitch, roll)
    
    # Step 2: Calculate d* values (magnitude of Q)
    d1 = np.linalg.norm(Q1)
    d2 = np.linalg.norm(Q2)
    
    # Step 3: Generate all possible (h,k,l) with d* < d*_max using ringdstar
    ringhkls = ringdstar(pa, polar_max, wl)
    
    # Step 4: Find possible (h,k,l) for Q1
    ha = []
    if ringhkls:
        dstar_keys = np.array(list(ringhkls.keys()))
        differences = np.abs(dstar_keys - d1)
        min_diff = np.min(differences)
        # Find all keys within a small tolerance of the minimum difference
        best_keys = dstar_keys[differences <= min_diff + 1e-12]
        for key in best_keys:
            ha.extend(ringhkls[key])
        # Sort the combined list lexicographically
        ha = sorted(ha)
    
    # Step 5: Find possible (h,k,l) for Q2
    hb = []
    if ringhkls:
        dstar_keys = np.array(list(ringhkls.keys()))
        differences = np.abs(dstar_keys - d2)
        min_diff = np.min(differences)
        best_keys = dstar_keys[differences <= min_diff + 1e-12]
        for key in best_keys:
            hb.extend(ringhkls[key])
        hb = sorted(hb)
    
    # Step 6: Print the possible (h,k,l) values
    print(f"Possible (h,k,l) for first reflection (pixel {p1}): {ha}")
    print(f"Possible (h,k,l) for second reflection (pixel {p2}): {hb}")
    
    return (ha, hb)



def Umat_p(pa, p1, p2, p3, b_c, det_d, p_s, wl, yaw, pitch, roll, z1, z2, z3, z_s, chi, phi, polar_max):
    '''Compute the U matrix which can best index $Q_3$ Bragg peak
    Input
    pa = (a,b,c,alpha,beta,gamma)
    a,b,c: the lengths a, b, and c of the three cell edges meeting at a vertex, float in the unit of angstrom
    alpha,beta,gamma: the angles alpha, beta, and gamma between those edges, float in the unit of degree
    p1: detector pixel (x1,y1), a tuple of two integer
    p2: detector pixel (x2,y2), a tuple of two integer
    p3: detector pixel (x3,y3), a tuple of two integer
    z1,z2,z3: frame number, integer
    b_c: incident beam center at detector pixel (xc,yc), a tuple of float
    det_d: sample distance to the detector, float in the unit of mm
    p_s: detector pixel size, and each pixel is a square, float in the unit of mm
    wl: X-ray wavelength, float in the unit of angstrom
    yaw,pitch,roll: rotation angles of the detector, float in the unit of degree
    z_s: step size in the \phi rotation, float in the unit of degree
    chi,phi: diffractometer angles, float in the unit of degree
    polar_max: maximum scattering angle, i.e. maximum angle between the x-ray beam axis
               and the powder ring, float in the unit of degree
    Output
    (best_U,best_H1,best_H2,best_H)): tuple (best_U,best_H1,best_H2,best_H).
                                      best_U: best U matrix, 3x3 orthogonal matrix, float;
                                      best_H1,best_H2: tuple (h,k,l) for which each element is an integer,
                                                       primary and secondary reflection for orientation
                                      best_H: indices of Q3 using best U matrix, 3x1 orthogonal matrix, float
    '''
    # Step 1: Compute Q1, Q2, Q3 from detector pixels and frames
    Q1 = q_cal_p(p1, b_c, det_d, p_s, wl, yaw, pitch, roll)
    Q2 = q_cal_p(p2, b_c, det_d, p_s, wl, yaw, pitch, roll)
    Q3 = q_cal_p(p3, b_c, det_d, p_s, wl, yaw, pitch, roll)
    
    # Step 2: Rotate Q2 and Q3 to the frame of Q1 (theta = z1*z_s)
    # Compute delta theta for Q2 relative to Q1
    delta_theta2_deg = (z2 - z1) * z_s
    delta_theta2_rad = np.radians(delta_theta2_deg)
    cos_dt2, sin_dt2 = np.cos(delta_theta2_rad), np.sin(delta_theta2_rad)
    R_y2 = np.array([
        [cos_dt2, 0, sin_dt2],
        [0, 1, 0],
        [-sin_dt2, 0, cos_dt2]
    ])
    Q2_rot = R_y2 @ Q2
    
    # Compute delta theta for Q3 relative to Q1
    delta_theta3_deg = (z3 - z1) * z_s
    delta_theta3_rad = np.radians(delta_theta3_deg)
    cos_dt3, sin_dt3 = np.cos(delta_theta3_rad), np.sin(delta_theta3_rad)
    R_y3 = np.array([
        [cos_dt3, 0, sin_dt3],
        [0, 1, 0],
        [-sin_dt3, 0, cos_dt3]
    ])
    Q3_rot = R_y3 @ Q3
    
    # Step 3: Compute angle between Q1 and Q2_rot
    dot_Q = np.dot(Q1.flatten(), Q2_rot.flatten())
    norm_Q1 = np.linalg.norm(Q1)
    norm_Q2rot = np.linalg.norm(Q2_rot)
    if norm_Q1 < 1e-12 or norm_Q2rot < 1e-12:
        return (None, None, None, None)
    cos_angle_Q = dot_Q / (norm_Q1 * norm_Q2rot)
    cos_angle_Q = np.clip(cos_angle_Q, -1.0, 1.0)
    angle_Q = np.arccos(cos_angle_Q)
    
    # Step 4: Get possible (h,k,l) for Q1 and Q2
    H1, H2 = hkl_pairs(pa, p1, p2, b_c, det_d, p_s, wl, yaw, pitch, roll, polar_max)
    if not H1 or not H2:
        return (None, None, None, None)
    
    # Step 5: Compute B matrix and its inverse
    B = Bmat(pa)
    B_inv = np.linalg.inv(B)
    
    # Step 6: Iterate over all valid (h1, h2) pairs
    best_distance = float('inf')
    best_U = None
    best_H1 = None
    best_H2 = None
    best_H = None
    
    for h1_tuple in H1:
        h1 = np.array([[h1_tuple[0]], [h1_tuple[1]], [h1_tuple[2]]])
        q1 = B @ h1
        norm_q1 = np.linalg.norm(q1)
        if norm_q1 < 1e-12:
            continue
        
        for h2_tuple in H2:
            h2 = np.array([[h2_tuple[0]], [h2_tuple[1]], [h2_tuple[2]]])
            q2 = B @ h2
            norm_q2 = np.linalg.norm(q2)
            if norm_q2 < 1e-12:
                continue
            
            # Compute angle between q1 and q2
            dot_q = np.dot(q1.flatten(), q2.flatten())
            cos_angle_q = dot_q / (norm_q1 * norm_q2)
            cos_angle_q = np.clip(cos_angle_q, -1.0, 1.0)
            angle_q = np.arccos(cos_angle_q)
            
            # Check if angles are close enough
            if np.abs(angle_q - angle_Q) > 1e-3:
                continue
            
            # Check if q1 and q2 are non-parallel
            cross_q = np.cross(q1.flatten(), q2.flatten())
            cross_q_norm = np.linalg.norm(cross_q)
            if cross_q_norm < 1e-12:
                continue
            
            # Construct T_c basis (crystal Cartesian system)
            t1c = q1 / norm_q1
            t3c = cross_q[:, np.newaxis] / cross_q_norm
            t2c = np.cross(t3c.flatten(), t1c.flatten())[:, np.newaxis]
            T_c = np.column_stack((t1c, t2c, t3c))
            
            # Construct T_g basis (lab frame at theta1)
            t1g = Q1 / norm_Q1
            cross_Q = np.cross(Q1.flatten(), Q2_rot.flatten())
            cross_Q_norm = np.linalg.norm(cross_Q)
            if cross_Q_norm < 1e-12:
                continue
            t3g = cross_Q[:, np.newaxis] / cross_Q_norm
            t2g = np.cross(t3g.flatten(), t1g.flatten())[:, np.newaxis]
            T_g = np.column_stack((t1g, t2g, t3g))
            
            # Compute orientation matrix U
            U = T_g @ T_c.T
            
            # Calculate (h,k,l) for Q3_rot using U and B
            UB_inv = B_inv @ U.T
            h3 = UB_inv @ Q3_rot
            
            # Calculate distance to nearest integer indices
            h3_int = np.round(h3).astype(int)
            diff = h3 - h3_int
            distance = np.linalg.norm(diff)
            
            # Update best results if current is better
            if distance < best_distance:
                best_distance = distance
                best_U = U.copy()
                best_H1 = h1_tuple
                best_H2 = h2_tuple 
                best_H = h3
    
    # Return best results or None if no valid pairs found
    if best_U is None:
        return (None, None, None, None)
    else:
        return (best_U, best_H1, best_H2, best_H)
