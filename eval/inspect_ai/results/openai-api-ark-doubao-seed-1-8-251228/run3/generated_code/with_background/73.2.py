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
