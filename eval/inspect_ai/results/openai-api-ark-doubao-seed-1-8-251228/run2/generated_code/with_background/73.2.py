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
