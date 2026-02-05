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
    
    # Calculate offset from beam center in millimeters
    dx = (x - xc) * p_s
    dy = (y - yc) * p_s
    
    # Convert rotation angles from degrees to radians
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)
    
    # Define individual rotation matrices (active rotation, right-handed)
    # Rotation around +z axis (yaw)
    cos_psi, sin_psi = np.cos(yaw_rad), np.sin(yaw_rad)
    Rz = np.array([
        [cos_psi, -sin_psi, 0.0],
        [sin_psi, cos_psi, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Rotation around +y axis (pitch)
    cos_theta, sin_theta = np.cos(pitch_rad), np.sin(pitch_rad)
    Ry = np.array([
        [cos_theta, 0.0, sin_theta],
        [0.0, 1.0, 0.0],
        [-sin_theta, 0.0, cos_theta]
    ])
    
    # Rotation around +x axis (roll)
    cos_phi, sin_phi = np.cos(roll_rad), np.sin(roll_rad)
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_phi, -sin_phi],
        [0.0, sin_phi, cos_phi]
    ])
    
    # Combined rotation matrix (sequence: yaw -> pitch -> roll)
    D = Rx @ Ry @ Rz
    
    # Offset vector from beam center to pixel in unrotated detector frame (lab coordinates)
    offset_unrot = np.array([0.0, -dx, -dy])
    
    # Rotate the offset vector to account for detector rotation
    offset_rot = D @ offset_unrot
    
    # Calculate pixel position in lab coordinates (mm)
    pixel_pos = np.array([det_d, 0.0, 0.0]) + offset_rot
    
    # Unit vector in the direction of the scattered beam
    s_magnitude = np.linalg.norm(pixel_pos)
    s = pixel_pos / s_magnitude
    
    # Compute wavevectors (k = 1/λ, unit: 1/Å)
    k = 1.0 / wl
    k_i = np.array([k, 0.0, 0.0])
    k_s = k * s
    
    # Calculate momentum transfer Q and reshape to 3x1 matrix
    Q = (k_s - k_i).reshape(3, 1)
    
    return Q



def Umat(t_c, t_g):
    '''Write down the orientation matrix which transforms from bases t_c to t_g
    Input
    t_c, tuple with three elements, each element is a 3x1 matrix, float
    t_g, tuple with three elements, each element is a 3x1 matrix, float
    Output
    U: 3x3 orthogonal matrix, float
    '''
    # Construct matrices with basis vectors as columns
    T_c = np.column_stack(t_c)
    T_g = np.column_stack(t_g)
    
    # Calculate orientation matrix using orthonormal basis property (inverse = transpose)
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
    z_s: step size in the rotation, float in the unit of degree
    chi,phi: diffractometer angles, float in the unit of degree
    Output
    q: 3x1 matrix, float
    '''
    # Step 1: Calculate B matrix using direct lattice parameters
    B = Bmat(pa)
    
    # Step 2: Compute crystal Cartesian coordinates for primary/secondary reflections
    H1_arr = np.array(H1).reshape(3, 1)
    q1 = B @ H1_arr
    H2_arr = np.array(H2).reshape(3, 1)
    q2 = B @ H2_arr
    
    # Step 3: Build orthonormal basis {t_c} from crystal Cartesian vectors
    # t1_c parallel to q1
    t1_c = q1 / np.linalg.norm(q1)
    # t3_c parallel to q1 × q2
    cross_q1q2 = np.cross(q1.flatten(), q2.flatten()).reshape(3, 1)
    t3_c = cross_q1q2 / np.linalg.norm(cross_q1q2)
    # t2_c completes right-handed orthonormal triple
    t2_c = np.cross(t3_c.flatten(), t1_c.flatten()).reshape(3, 1)
    t_c = (t1_c, t2_c, t3_c)
    
    # Step 4: Convert diffractometer angles to radians
    chi_rad = np.radians(chi)
    phi_rad = np.radians(phi)
    
    # Step 5: Precompute common rotation matrices for diffractometer
    # Rotation around +x axis (chi)
    cos_chi, sin_chi = np.cos(chi_rad), np.sin(chi_rad)
    Rx_chi = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_chi, -sin_chi],
        [0.0, sin_chi, cos_chi]
    ])
    # Rotation around +z axis (phi)
    cos_phi, sin_phi = np.cos(phi_rad), np.sin(phi_rad)
    Rz_phi = np.array([
        [cos_phi, -sin_phi, 0.0],
        [sin_phi, cos_phi, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Step 6: Calculate Q1 (pre-rotation momentum transfer for primary reflection)
    theta1_deg = z1 * z_s
    theta1_rad = np.radians(theta1_deg)
    cos_theta1, sin_theta1 = np.cos(theta1_rad), np.sin(theta1_rad)
    # Rotation around -y axis (theta1)
    Ry_minus_theta1 = np.array([
        [cos_theta1, 0.0, -sin_theta1],
        [0.0, 1.0, 0.0],
        [sin_theta1, 0.0, cos_theta1]
    ])
    G1 = Ry_minus_theta1 @ Rx_chi @ Rz_phi
    # Measured momentum transfer in lab coordinates
    Q1_measured = q_cal_p(p1, b_c, det_d, p_s, wl, yaw, pitch, roll)
    # Rotate back to pre-rotation orientation
    Q1 = G1.T @ Q1_measured
    
    # Step 7: Calculate Q2 (pre-rotation momentum transfer for secondary reflection)
    theta2_deg = z2 * z_s
    theta2_rad = np.radians(theta2_deg)
    cos_theta2, sin_theta2 = np.cos(theta2_rad), np.sin(theta2_rad)
    Ry_minus_theta2 = np.array([
        [cos_theta2, 0.0, -sin_theta2],
        [0.0, 1.0, 0.0],
        [sin_theta2, 0.0, cos_theta2]
    ])
    G2 = Ry_minus_theta2 @ Rx_chi @ Rz_phi
    Q2_measured = q_cal_p(p2, b_c, det_d, p_s, wl, yaw, pitch, roll)
    Q2 = G2.T @ Q2_measured
    
    # Step 8: Build orthonormal basis {t_g} from pre-rotation momentum transfers
    t1_g = Q1 / np.linalg.norm(Q1)
    cross_Q1Q2 = np.cross(Q1.flatten(), Q2.flatten()).reshape(3, 1)
    t3_g = cross_Q1Q2 / np.linalg.norm(cross_Q1Q2)
    t2_g = np.cross(t3_g.flatten(), t1_g.flatten()).reshape(3, 1)
    t_g = (t1_g, t2_g, t3_g)
    
    # Step 9: Calculate orientation matrix U
    U = Umat(t_c, t_g)
    
    # Step 10: Process input pixel to get (h,k,l)
    # Calculate diffractometer rotation matrix for current frame z
    theta_z_deg = z * z_s
    theta_z_rad = np.radians(theta_z_deg)
    cos_theta_z, sin_theta_z = np.cos(theta_z_rad), np.sin(theta_z_rad)
    Ry_minus_thetaz = np.array([
        [cos_theta_z, 0.0, -sin_theta_z],
        [0.0, 1.0, 0.0],
        [sin_theta_z, 0.0, cos_theta_z]
    ])
    G_z = Ry_minus_thetaz @ Rx_chi @ Rz_phi
    # Measured momentum transfer for input pixel
    Q_measured = q_cal_p(p, b_c, det_d, p_s, wl, yaw, pitch, roll)
    # Rotate back to pre-rotation orientation
    Q_prime = G_z.T @ Q_measured
    
    # Step 11: Convert to reciprocal lattice coordinates (h,k,l)
    UB = U @ B
    inv_UB = np.linalg.inv(UB)
    hkl = inv_UB @ Q_prime
    
    return hkl


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
    # Calculate B matrix using provided Bmat function
    B = Bmat(pa)
    
    # Compute d*_max using Bragg's law
    polar_max_rad = np.radians(polar_max)
    dstar_max = 2 * np.sin(polar_max_rad / 2) / wl
    
    # Construct quadratic form matrix for squared d* calculation
    C = B.T @ B
    
    # Determine maximum range of h, k, l using minimal eigenvalue of C
    eig_vals = np.linalg.eigvalsh(C)
    lambda_min = np.min(eig_vals)
    
    if lambda_min <= 1e-12:
        max_abs = 0
    else:
        max_abs = int(np.floor(dstar_max / np.sqrt(lambda_min))) + 1
    
    ringhkls = {}
    
    # Iterate over all possible integer triples (h, k, l)
    for h in range(-max_abs, max_abs + 1):
        for k in range(-max_abs, max_abs + 1):
            for l in range(-max_abs, max_abs + 1):
                # Skip reciprocal lattice origin (not a Bragg reflection)
                if h == 0 and k == 0 and l == 0:
                    continue
                
                # Calculate squared d* using quadratic form for efficiency
                v = np.array([h, k, l])
                dstar_sq = v @ C @ v
                
                # Skip if d* >= d*_max (using squared values to avoid sqrt)
                if dstar_sq >= dstar_max ** 2 - 1e-12:
                    continue
                
                dstar = np.sqrt(dstar_sq)
                
                # Check if this d* is already in the dictionary (within tolerance)
                found = False
                for key in list(ringhkls.keys()):
                    if np.isclose(dstar, key, atol=1e-9):
                        ringhkls[key].append((h, k, l))
                        found = True
                        break
                if not found:
                    ringhkls[dstar] = [(h, k, l)]
    
    # Sort each list of (h,k,l) tuples lexicographically
    for key in ringhkls:
        ringhkls[key] = sorted(ringhkls[key])
    
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
    # Step 1: Calculate momentum transfers Q1 and Q2 for the two detector pixels
    Q1 = q_cal_p(p1, b_c, det_d, p_s, wl, yaw, pitch, roll)
    Q2 = q_cal_p(p2, b_c, det_d, p_s, wl, yaw, pitch, roll)
    
    # Step 2: Compute d* values (magnitude of Q vectors)
    d1 = np.linalg.norm(Q1)
    d2 = np.linalg.norm(Q2)
    
    # Step 3: Generate all possible (h,k,l) and their corresponding d* values from lattice parameters
    ringhkls = ringdstar(pa, polar_max, wl)
    
    # Handle edge case where no valid (h,k,l) exist within polar_max
    if not ringhkls:
        return ([], [])
    
    # Step 4: Extract computed d* values from the ringhkls dictionary
    dstar_keys = np.array(list(ringhkls.keys()))
    
    # Step 5: Find closest computed d* to measured d1 and get corresponding (h,k,l) list
    diff1 = np.abs(dstar_keys - d1)
    closest_idx1 = np.argmin(diff1)
    closest_d1 = dstar_keys[closest_idx1]
    ha = ringhkls[closest_d1]
    
    # Step 6: Find closest computed d* to measured d2 and get corresponding (h,k,l) list
    diff2 = np.abs(dstar_keys - d2)
    closest_idx2 = np.argmin(diff2)
    closest_d2 = dstar_keys[closest_idx2]
    hb = ringhkls[closest_d2]
    
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
    # Step 1: Calculate B matrix
    B = Bmat(pa)
    
    # Step 2: Compute measured momentum transfers for Q1, Q2, Q3
    Q1_measured = q_cal_p(p1, b_c, det_d, p_s, wl, yaw, pitch, roll)
    Q2_measured = q_cal_p(p2, b_c, det_d, p_s, wl, yaw, pitch, roll)
    Q3_measured = q_cal_p(p3, b_c, det_d, p_s, wl, yaw, pitch, roll)
    
    # Step 3: Compute diffractometer rotation matrices G1, G2, G3
    # Convert angles to radians
    chi_rad = np.radians(chi)
    phi_rad = np.radians(phi)
    
    # Common rotation matrices (Rx_chi and Rz_phi are same for all z)
    cos_chi = np.cos(chi_rad)
    sin_chi = np.sin(chi_rad)
    Rx_chi = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_chi, -sin_chi],
        [0.0, sin_chi, cos_chi]
    ])
    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)
    Rz_phi = np.array([
        [cos_phi, -sin_phi, 0.0],
        [sin_phi, cos_phi, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Compute G1 for z1
    theta1_deg = z1 * z_s
    theta1_rad = np.radians(theta1_deg)
    cos_theta1 = np.cos(theta1_rad)
    sin_theta1 = np.sin(theta1_rad)
    Ry_minus_theta1 = np.array([
        [cos_theta1, 0.0, -sin_theta1],
        [0.0, 1.0, 0.0],
        [sin_theta1, 0.0, cos_theta1]
    ])
    G1 = Ry_minus_theta1 @ Rx_chi @ Rz_phi
    
    # Compute G2 for z2
    theta2_deg = z2 * z_s
    theta2_rad = np.radians(theta2_deg)
    cos_theta2 = np.cos(theta2_rad)
    sin_theta2 = np.sin(theta2_rad)
    Ry_minus_theta2 = np.array([
        [cos_theta2, 0.0, -sin_theta2],
        [0.0, 1.0, 0.0],
        [sin_theta2, 0.0, cos_theta2]
    ])
    G2 = Ry_minus_theta2 @ Rx_chi @ Rz_phi
    
    # Compute G3 for z3
    theta3_deg = z3 * z_s
    theta3_rad = np.radians(theta3_deg)
    cos_theta3 = np.cos(theta3_rad)
    sin_theta3 = np.sin(theta3_rad)
    Ry_minus_theta3 = np.array([
        [cos_theta3, 0.0, -sin_theta3],
        [0.0, 1.0, 0.0],
        [sin_theta3, 0.0, cos_theta3]
    ])
    G3 = Ry_minus_theta3 @ Rx_chi @ Rz_phi
    
    # Step 4: Compute Q1', Q2', Q3' (pre-rotation momentum transfers)
    Q1_prime = G1.T @ Q1_measured
    Q2_prime = G2.T @ Q2_measured
    Q3_prime = G3.T @ Q3_measured
    
    # Step 5: Compute angle between Q1' and Q2' (theta_meas)
    norm_Q1p = np.linalg.norm(Q1_prime)
    norm_Q2p = np.linalg.norm(Q2_prime)
    if norm_Q1p < 1e-12 or norm_Q2p < 1e-12:
        return (None, None, None, None)
    cos_theta_meas = (Q1_prime.T @ Q2_prime).item() / (norm_Q1p * norm_Q2p)
    cos_theta_meas = np.clip(cos_theta_meas, -1.0, 1.0)
    theta_meas = np.arccos(cos_theta_meas)
    
    # Step 6: Get possible (h,k,l) pairs for Q1 and Q2
    H1_list, H2_list = hkl_pairs(pa, p1, p2, b_c, det_d, p_s, wl, yaw, pitch, roll, polar_max)
    
    # Step 7: Find valid (H1, H2) pairs with matching angles
    valid_pairs = []
    for H1 in H1_list:
        for H2 in H2_list:
            H1_arr = np.array(H1).reshape(3, 1)
            q1 = B @ H1_arr
            H2_arr = np.array(H2).reshape(3, 1)
            q2 = B @ H2_arr
            
            norm_q1 = np.linalg.norm(q1)
            norm_q2 = np.linalg.norm(q2)
            if norm_q1 < 1e-12 or norm_q2 < 1e-12:
                continue
            
            cos_theta_c = (q1.T @ q2).item() / (norm_q1 * norm_q2)
            cos_theta_c = np.clip(cos_theta_c, -1.0, 1.0)
            theta_c = np.arccos(cos_theta_c)
            
            if np.isclose(theta_c, theta_meas, atol=1e-3):
                valid_pairs.append((H1, H2))
    
    if not valid_pairs:
        return (None, None, None, None)
    
    # Step 8: Evaluate each valid pair to find best U
    candidates = []
    for H1, H2 in valid_pairs:
        # Compute q1 and q2
        H1_arr = np.array(H1).reshape(3, 1)
        q1 = B @ H1_arr
        H2_arr = np.array(H2).reshape(3, 1)
        q2 = B @ H2_arr
        
        # Build orthonormal basis t_c
        t1_c = q1 / np.linalg.norm(q1)
        cross_q1q2 = np.cross(q1.flatten(), q2.flatten()).reshape(3, 1)
        norm_cross = np.linalg.norm(cross_q1q2)
        if norm_cross < 1e-12:
            continue
        t3_c = cross_q1q2 / norm_cross
        t2_c = np.cross(t3_c.flatten(), t1_c.flatten()).reshape(3, 1)
        t_c = (t1_c, t2_c, t3_c)
        
        # Build orthonormal basis t_g
        t1_g = Q1_prime / np.linalg.norm(Q1_prime)
        cross_Q1Q2 = np.cross(Q1_prime.flatten(), Q2_prime.flatten()).reshape(3, 1)
        norm_cross_g = np.linalg.norm(cross_Q1Q2)
        if norm_cross_g < 1e-12:
            continue
        t3_g = cross_Q1Q2 / norm_cross_g
        t2_g = np.cross(t3_g.flatten(), t1_g.flatten()).reshape(3, 1)
        t_g = (t1_g, t2_g, t3_g)
        
        # Compute U matrix
        try:
            U = Umat(t_c, t_g)
        except Exception:
            continue
        
        # Compute UB and its inverse
        UB = U @ B
        try:
            inv_UB = np.linalg.inv(UB)
        except np.linalg.LinAlgError:
            continue
        
        # Compute hkl candidate for Q3'
        hkl_candidate = inv_UB @ Q3_prime
        
        # Compute nearest integer triple and distance
        hkl_int = np.round(hkl_candidate).astype(int).flatten()
        hkl_int_arr = hkl_int.reshape(3, 1)
        distance = np.linalg.norm(hkl_candidate - hkl_int_arr)
        
        # Store candidate
        candidates.append((distance, U, H1, H2, hkl_candidate))
    
    if not candidates:
        return (None, None, None, None)
    
    # Step 9: Select best candidate with smallest distance
    candidates.sort(key=lambda x: x[0])
    best_distance, best_U, best_H1, best_H2, best_H = candidates[0]
    
    return (best_U, best_H1, best_H2, best_H)



def auto_index(pa, px, py, b_c, det_d, p_s, wl, yaw, pitch, roll, z, z_s, chi, phi, polar_max):
    '''Index all the Bragg peaks in the list
    Input
    crystal structure:
    pa = (a,b,c,alpha,beta,gamma)
    a,b,c: the lengths a, b, and c of the three cell edges meeting at a vertex, float in the unit of angstrom
    alpha,beta,gamma: the angles alpha, beta, and gamma between those edges, float in the unit of degree
    list of Bragg peaks to be indexed:
    px,py: detector pixel (px,py); px,py is a list of integer
    z: frame number, a list of integer
    instrument configuration:
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
    HKL: indices of Bragg peaks, a list, each element is a tuple (h,k,l).
         The values of h, k, and l are rounded to two decimal places.
    '''
    # Validate input list consistency and minimum peak requirement
    if len(px) != len(py) or len(px) != len(z):
        return []
    num_peaks = len(px)
    if num_peaks < 3:
        return []
    
    # Extract first three peaks for orientation determination
    p1 = (px[0], py[0])
    z1 = z[0]
    p2 = (px[1], py[1])
    z2 = z[1]
    p3 = (px[2], py[2])
    z3 = z[2]
    
    # Retrieve best orientation matrix using the first three peaks
    best_U, _, _, _ = Umat_p(pa, p1, p2, p3, b_c, det_d, p_s, wl, yaw, pitch, roll, z1, z2, z3, z_s, chi, phi, polar_max)
    
    # Return empty list if no valid orientation matrix found
    if best_U is None:
        return []
    
    # Precompute fixed matrices for indexing
    B = Bmat(pa)
    UB = best_U @ B
    inv_UB = np.linalg.inv(UB)
    
    # Precompute diffractometer rotation matrices for fixed chi and phi
    chi_rad = np.radians(chi)
    phi_rad = np.radians(phi)
    
    # Rotation matrix around +x axis (chi)
    cos_chi = np.cos(chi_rad)
    sin_chi = np.sin(chi_rad)
    Rx_chi = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_chi, -sin_chi],
        [0.0, sin_chi, cos_chi]
    ])
    
    # Rotation matrix around +z axis (phi)
    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)
    Rz_phi = np.array([
        [cos_phi, -sin_phi, 0.0],
        [sin_phi, cos_phi, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    HKL = []
    for i in range(num_peaks):
        # Get current peak coordinates and frame
        x_pixel = px[i]
        y_pixel = py[i]
        current_z = z[i]
        
        # Calculate measured momentum transfer in lab coordinates
        Q_measured = q_cal_p((x_pixel, y_pixel), b_c, det_d, p_s, wl, yaw, pitch, roll)
        
        # Compute diffractometer rotation matrix for current frame
        theta_deg = current_z * z_s
        theta_rad = np.radians(theta_deg)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        
        # Rotation around -y axis (theta)
        Ry_minus_theta = np.array([
            [cos_theta, 0.0, -sin_theta],
            [0.0, 1.0, 0.0],
            [sin_theta, 0.0, cos_theta]
        ])
        
        G = Ry_minus_theta @ Rx_chi @ Rz_phi
        
        # Rotate measured Q back to pre-rotation orientation
        Q_prime = G.T @ Q_measured
        
        # Calculate reciprocal lattice coordinates
        hkl = inv_UB @ Q_prime
        
        # Round to two decimal places and format as tuple
        h = round(float(hkl[0]), 2)
        k = round(float(hkl[1]), 2)
        l = round(float(hkl[2]), 2)
        HKL.append((h, k, l))
    
    return HKL
