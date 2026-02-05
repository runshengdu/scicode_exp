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
