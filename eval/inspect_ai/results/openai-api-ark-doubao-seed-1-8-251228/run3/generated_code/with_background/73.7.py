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
    
    # Construct direct lattice vectors in Cartesian coordinates
    a1 = np.array([a, 0.0, 0.0])
    a2_x = b * np.cos(gamma_rad)
    a2_y = b * np.sin(gamma_rad)
    a2 = np.array([a2_x, a2_y, 0.0])
    
    a3_x = c * np.cos(beta_rad)
    numerator_y = np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)
    denominator_y = np.sin(gamma_rad)
    a3_y = c * numerator_y / denominator_y
    
    # Calculate z-component of a3, handle numerical precision
    a3_z_sq = c**2 - a3_x**2 - a3_y**2
    a3_z = np.sqrt(np.maximum(a3_z_sq, 0.0))
    a3 = np.array([a3_x, a3_y, a3_z])
    
    # Compute unit cell volume via scalar triple product
    V = np.dot(a1, np.cross(a2, a3))
    
    # Calculate reciprocal lattice vectors
    b1 = np.cross(a2, a3) / V
    b2 = np.cross(a3, a1) / V
    b3 = np.cross(a1, a2) / V
    
    # Form B matrix with reciprocal vectors as columns
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
    # Convert angles from degrees to radians
    psi_rad = np.radians(yaw)
    theta_rad = np.radians(pitch)
    phi_rad = np.radians(roll)
    
    # Calculate pixel offsets in detector coordinates (mm)
    dx_pixel = (p[0] - b_c[0]) * p_s
    dy_pixel = (p[1] - b_c[1]) * p_s
    
    # Vector from sample to detector pixel in detector coordinates (mm)
    vec_det = np.array([dx_pixel, dy_pixel, det_d], dtype=np.float64)
    
    # Default detector axes in lab coordinate system
    x0_det = np.array([0.0, -1.0, 0.0])
    y0_det = np.array([0.0, 0.0, -1.0])
    z0_det = np.array([1.0, 0.0, 0.0])
    M0 = np.column_stack((x0_det, y0_det, z0_det))
    
    # Rotation matrices for yaw, pitch, roll (extrinsic rotations around lab axes)
    cos_psi, sin_psi = np.cos(psi_rad), np.sin(psi_rad)
    Rz = np.array([
        [cos_psi, -sin_psi, 0.0],
        [sin_psi, cos_psi, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    cos_theta, sin_theta = np.cos(theta_rad), np.sin(theta_rad)
    Ry = np.array([
        [cos_theta, 0.0, sin_theta],
        [0.0, 1.0, 0.0],
        [-sin_theta, 0.0, cos_theta]
    ], dtype=np.float64)
    
    cos_phi, sin_phi = np.cos(phi_rad), np.sin(phi_rad)
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_phi, -sin_phi],
        [0.0, sin_phi, cos_phi]
    ], dtype=np.float64)
    
    # Detector rotation matrix D
    D = Rx @ Ry @ Rz
    
    # Transformation matrix from detector coordinates to lab coordinates
    M = D @ M0
    
    # Vector from sample to detector pixel in lab coordinates (mm)
    vec_lab = M @ vec_det
    
    # Unit vector in the direction of scattered beam
    dist = np.linalg.norm(vec_lab)
    if dist == 0.0:
        u_s = np.zeros(3, dtype=np.float64)
    else:
        u_s = vec_lab / dist
    
    # Calculate k vectors (k = 1/lambda)
    k_mag = 1.0 / wl  # 1/angstrom
    k_i = k_mag * np.array([1.0, 0.0, 0.0], dtype=np.float64)  # Incident wavevector (along +x lab)
    k_s = k_mag * u_s  # Scattered wavevector
    
    # Momentum transfer Q = k_s - k_i
    Q = k_s - k_i
    
    # Convert to 3x1 matrix
    Q = Q.reshape(3, 1)
    
    return Q



def u_triple_p(pa, H1, H2, p1, p2, b_c, det_d, p_s, wl, yaw, pitch, roll, z1, z2, z_s, chi, phi):
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
    yaw,pitch,roll: rotation angles of the detector, float in the unit of degree
    z1,z2: frame number, integer
    z_s: step size in the \phi rotation, float in the unit of degree
    chi,phi: diffractometer angles, float in the unit of degree
    Output
    t_c_t_g: tuple (t_c,t_g), t_c = (t1c,t2c,t3c) and t_g = (t1g,t2g,t3g).
    Each element inside t_c and t_g is a 3x1 matrix, float
    '''
    # Calculate B matrix using provided Bmat function
    B = Bmat(pa)
    
    # Compute q1 and q2 (reciprocal lattice vectors in crystal Cartesian frame)
    H1_col = np.array(H1).reshape(3, 1)
    H2_col = np.array(H2).reshape(3, 1)
    q1 = B @ H1_col
    q2 = B @ H2_col
    
    # Construct t_c orthogonal unit-vector triple
    # t1c: unit vector along q1
    norm_q1 = np.linalg.norm(q1)
    t1c = q1 / norm_q1
    
    # t3c: unit vector along q1 × q2
    cross_c = np.cross(q1.flatten(), q2.flatten()).reshape(3, 1)
    norm_cross_c = np.linalg.norm(cross_c)
    t3c = cross_c / norm_cross_c
    
    # t2c: unit vector orthogonal to t1c and t3c, forming right-handed triple
    t2c = np.cross(t3c.flatten(), t1c.flatten()).reshape(3, 1)
    t_c = (t1c, t2c, t3c)
    
    # Calculate theta angles for each frame (degrees)
    theta1 = z1 * z_s
    theta2 = z2 * z_s
    
    # Compute lab-frame momentum transfers from observed peaks
    Q_lab_1 = q_cal_p(p1, b_c, det_d, p_s, wl, yaw, pitch, roll)
    Q_lab_2 = q_cal_p(p2, b_c, det_d, p_s, wl, yaw, pitch, roll)
    
    # Convert diffractometer angles to radians
    phi_rad = np.radians(phi)
    chi_rad = np.radians(chi)
    theta1_rad = np.radians(theta1)
    theta2_rad = np.radians(theta2)
    
    # Define basic rotation matrices
    # Rotation around z-axis (phi)
    cos_phi, sin_phi = np.cos(phi_rad), np.sin(phi_rad)
    Rz = np.array([
        [cos_phi, -sin_phi, 0.0],
        [sin_phi, cos_phi, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    # Rotation around x-axis (chi)
    cos_chi, sin_chi = np.cos(chi_rad), np.sin(chi_rad)
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_chi, -sin_chi],
        [0.0, sin_chi, cos_chi]
    ], dtype=np.float64)
    
    # Compute full rotation matrices for each frame
    # Rotation around y-axis (-theta1)
    cos_theta1, sin_theta1 = np.cos(theta1_rad), np.sin(theta1_rad)
    Ry1 = np.array([
        [cos_theta1, 0.0, -sin_theta1],
        [0.0, 1.0, 0.0],
        [sin_theta1, 0.0, cos_theta1]
    ], dtype=np.float64)
    G1 = Ry1 @ Rx @ Rz
    
    # Rotation around y-axis (-theta2)
    cos_theta2, sin_theta2 = np.cos(theta2_rad), np.sin(theta2_rad)
    Ry2 = np.array([
        [cos_theta2, 0.0, -sin_theta2],
        [0.0, 1.0, 0.0],
        [sin_theta2, 0.0, cos_theta2]
    ], dtype=np.float64)
    G2 = Ry2 @ Rx @ Rz
    
    # Convert lab-frame momentum transfers to crystal frame (before rotation)
    Q1 = G1.T @ Q_lab_1
    Q2 = G2.T @ Q_lab_2
    
    # Construct t_g orthogonal unit-vector triple
    # t1g: unit vector along Q1
    norm_Q1 = np.linalg.norm(Q1)
    t1g = Q1 / norm_Q1
    
    # t3g: unit vector along Q1 × Q2
    cross_g = np.cross(Q1.flatten(), Q2.flatten()).reshape(3, 1)
    norm_cross_g = np.linalg.norm(cross_g)
    t3g = cross_g / norm_cross_g
    
    # t2g: unit vector orthogonal to t1g and t3g, forming right-handed triple
    t2g = np.cross(t3g.flatten(), t1g.flatten()).reshape(3, 1)
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
    # Construct matrices T_c and T_g with columns as the basis vectors
    T_c = np.column_stack(t_c)
    T_g = np.column_stack(t_g)
    
    # Calculate U as T_g multiplied by the transpose of T_c
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
    q: 3x1 orthogonal matrix, float
    '''
    # Step 1: Calculate lab frame momentum transfer Q
    Q = q_cal_p(p, b_c, det_d, p_s, wl, yaw, pitch, roll)
    
    # Step 2: Calculate diffractometer rotation matrix G and its inverse
    # Compute theta angle for current frame
    theta_deg = z * z_s
    # Convert angles to radians
    theta_rad = np.radians(theta_deg)
    phi_rad = np.radians(phi)
    chi_rad = np.radians(chi)
    
    # Rotation matrices
    # Rz: rotation around z-axis by phi
    cos_phi, sin_phi = np.cos(phi_rad), np.sin(phi_rad)
    Rz = np.array([
        [cos_phi, -sin_phi, 0.0],
        [sin_phi, cos_phi, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    # Rx: rotation around x-axis by chi
    cos_chi, sin_chi = np.cos(chi_rad), np.sin(chi_rad)
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_chi, -sin_chi],
        [0.0, sin_chi, cos_chi]
    ], dtype=np.float64)
    
    # Ry: rotation around y-axis by -theta (equivalent to rotation around -y by theta)
    cos_theta, sin_theta = np.cos(theta_rad), np.sin(theta_rad)
    Ry = np.array([
        [cos_theta, 0.0, -sin_theta],
        [0.0, 1.0, 0.0],
        [sin_theta, 0.0, cos_theta]
    ], dtype=np.float64)
    
    # Compute G matrix: G = Ry @ Rx @ Rz (rotation sequence phi -> chi -> theta)
    G = Ry @ Rx @ Rz
    # Inverse of orthogonal matrix is its transpose
    G_inv = G.T
    
    # Transform lab frame Q to crystal's initial orientation Q'
    Q_prime = G_inv @ Q
    
    # Step 3: Calculate orientation matrix U and reciprocal lattice matrix B
    # Get orthogonal basis triples from reflection data
    t_c, t_g = u_triple_p(pa, H1, H2, p1, p2, b_c, det_d, p_s, wl, yaw, pitch, roll, z1, z2, z_s, chi, phi)
    # Compute orientation matrix U
    U = Umat(t_c, t_g)
    # Compute B matrix from lattice parameters
    B = Bmat(pa)
    
    # Step 4: Calculate (h,k,l) = (U*B)^-1 @ Q'
    UB = U @ B
    inv_UB = np.linalg.inv(UB)
    q = inv_UB @ Q_prime
    
    return q


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
    # Calculate B matrix using provided function
    B = Bmat(pa)
    
    # Compute d*_max using Bragg's law derivation
    polar_max_rad = np.radians(polar_max)
    dstar_max = 2 * np.sin(polar_max_rad / 2) / wl
    
    # Calculate magnitudes of reciprocal lattice vectors
    b1_mag = np.linalg.norm(B[:, 0])
    b2_mag = np.linalg.norm(B[:, 1])
    b3_mag = np.linalg.norm(B[:, 2])
    
    # Determine maximum absolute values for h, k, l indices
    h_max = int(np.floor(dstar_max / b1_mag)) if b1_mag > 0 else 0
    k_max = int(np.floor(dstar_max / b2_mag)) if b2_mag > 0 else 0
    l_max = int(np.floor(dstar_max / b3_mag)) if b3_mag > 0 else 0
    
    # Ensure non-negative maximum indices
    h_max = max(h_max, 0)
    k_max = max(k_max, 0)
    l_max = max(l_max, 0)
    
    ringhkls = {}
    epsilon = 1e-8  # Tolerance for grouping numerically close d* values
    
    # Iterate through all possible Miller indices
    for h in range(-h_max, h_max + 1):
        for k in range(-k_max, k_max + 1):
            for l in range(-l_max, l_max + 1):
                # Skip the direct beam (0,0,0) reflection
                if h == 0 and k == 0 and l == 0:
                    continue
                
                # Compute reciprocal lattice vector magnitude (d*)
                q = B @ np.array([h, k, l]).reshape(3, 1)
                dstar = np.linalg.norm(q)
                
                # Skip if d* exceeds or equals maximum allowed value
                if dstar >= dstar_max:
                    continue
                
                # Find existing key with similar d* value
                matched_key = None
                for key in ringhkls:
                    if abs(dstar - key) < epsilon:
                        matched_key = key
                        break
                
                # Add indices to appropriate list
                if matched_key is not None:
                    ringhkls[matched_key].append((h, k, l))
                else:
                    ringhkls[dstar] = [(h, k, l)]
    
    # Sort each list of Miller indices lexicographically
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
    # Calculate momentum transfer Q for each detector pixel
    Q1 = q_cal_p(p1, b_c, det_d, p_s, wl, yaw, pitch, roll)
    Q2 = q_cal_p(p2, b_c, det_d, p_s, wl, yaw, pitch, roll)
    
    # Compute d* values as the magnitude of the momentum transfer vectors
    dstar1 = np.linalg.norm(Q1)
    dstar2 = np.linalg.norm(Q2)
    
    # Retrieve all possible (h,k,l) reflections and their corresponding d* values
    ringhkls = ringdstar(pa, polar_max, wl)
    
    epsilon = 1e-8  # Tolerance for matching d* values, consistent with ringdstar
    ha = []
    # Find matching d* for the first reflection
    for key in ringhkls:
        if abs(key - dstar1) < epsilon:
            ha = ringhkls[key]
            break
    
    hb = []
    # Find matching d* for the second reflection
    for key in ringhkls:
        if abs(key - dstar2) < epsilon:
            hb = ringhkls[key]
            break
    
    return (ha, hb)
