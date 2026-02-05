import numpy as np

def hopping_mk(d, dz, v_p0=-2.7, v_s0=0.48, b=1.17, a0=2.68, d0=6.33):
    '''Parameterization from Moon and Koshino, Phys. Rev. B 85, 195458 (2012).
    Args:
        d: distance between two atoms (unit b,a.u.), float
        dz: out-of-plane distance between two atoms (unit b,a.u.), float
        v_p0: transfer integral between the nearest-neighbor atoms of monolayer graphene, MK parameter, float,unit eV
        v_s0: interlayer transfer integral between vertically located atoms, MK parameter, float,unit eV
        b: 1/b is the decay length of the transfer integral, MK parameter, float, unit (b,a.u.)^-1
        a0: nearest-neighbor atom distance of the monolayer graphene, MK parameter, float, unit (b,a.u.)
        d0: interlayer distance, MK parameter, float, (b,a.u.)
    Return:
        hopping: -t, float, eV
    '''
    # Calculate ppi and psigma potentials
    v_pppi = v_p0 * np.exp(-b * (d - a0))
    v_pps = v_s0 * np.exp(-b * (d - d0))
    
    # Calculate the squared out-of-plane distance ratio
    dz_over_d_sq = (dz / d) ** 2
    
    # Compute the hopping parameter
    hopping = v_pppi * (1 - dz_over_d_sq) + v_pps * dz_over_d_sq
    
    return hopping


def mk(latvecs, basis, di, dj, ai, aj):
    '''Evaluate the Moon and Koshino hopping parameters Phys. Rev. B 85, 195458 (2012).
    Args:
        latvecs (np.array): lattice vectors of shape (3, 3) in bohr
        basis (np.array): atomic positions of shape (natoms, 3) in bohr; natoms: number of atoms within a unit cell
        di, dj (np.array): list of displacement indices for the hopping
        ai, aj (np.array): list of atomic basis indices for the hopping
    Return
        hopping (np.array): a list with the same length as di
    '''
    # Get atomic positions for each hopping pair
    pos_a = basis[ai]
    pos_b = basis[aj]
    
    # Calculate unit cell displacement difference
    delta_unit = di - dj
    
    # Compute lattice displacement vector
    delta_lat = delta_unit @ latvecs
    
    # Total vector between the two atoms
    delta = (pos_a - pos_b) + delta_lat
    
    # Calculate interatomic distance and out-of-plane component
    d = np.linalg.norm(delta, axis=1)
    dz = delta[:, 2]
    
    # Compute hopping using Moon-Koshino parameterization
    hop = hopping_mk(d, dz)
    
    return hop



def ham_eig(k_input, latvecs, basis):
    '''Calculate the eigenvalues for a given k-point (k-point is in reduced coordinates)
    Args:
        k_input (np.array): (kx, ky)
        latvecs (np.array): lattice vectors of shape (3, 3) in bohr
        basis (np.array): atomic positions of shape (natoms, 3) in bohr
    Returns:
        eigval: numpy array of floats, sorted array of eigenvalues
    '''
    natoms = basis.shape[0]
    if natoms == 0:
        return np.array([])
    
    # Extract real-space lattice vectors
    l1, l2, l3 = latvecs[0], latvecs[1], latvecs[2]
    
    # Calculate unit cell volume and reciprocal lattice vectors
    vol = np.dot(l1, np.cross(l2, l3))
    if abs(vol) < 1e-12:
        raise ValueError("Lattice vectors are linearly dependent (unit cell volume is zero).")
    
    b1 = 2 * np.pi * np.cross(l2, l3) / vol
    b2 = 2 * np.pi * np.cross(l3, l1) / vol
    
    # Convert reduced k-coordinates to reciprocal space vector (rad/bohr)
    kx, ky = k_input
    k_vec = kx * b1 + ky * b2
    
    # Initialize Hamiltonian matrix
    H = np.zeros((natoms, natoms), dtype=np.complex128)
    
    # Parameters for hopping calculation
    tol = 1e-8  # Tolerance for negligible hopping contributions
    max_d = 22.0  # Maximum interatomic distance to consider (bohr)
    
    for alpha in range(natoms):
        for beta in range(natoms):
            delta0 = basis[alpha] - basis[beta]
            D = np.linalg.norm(delta0)
            
            # Calculate maximum displacement indices for each lattice direction
            l1_len = np.linalg.norm(l1)
            l2_len = np.linalg.norm(l2)
            l3_len = np.linalg.norm(l3)
            
            max_n1 = int(np.ceil((max_d + D) / l1_len)) if l1_len > 1e-12 else 0
            max_n2 = int(np.ceil((max_d + D) / l2_len)) if l2_len > 1e-12 else 0
            max_n3 = int(np.ceil((max_d + D) / l3_len)) if l3_len > 1e-12 else 0
            
            # Iterate over all relevant displacement vectors
            for n1 in range(-max_n1, max_n1 + 1):
                for n2 in range(-max_n2, max_n2 + 1):
                    for n3 in range(-max_n3, max_n3 + 1):
                        R_unit = np.array([n1, n2, n3], dtype=np.int64)
                        R_lat = R_unit @ latvecs
                        delta = delta0 + R_lat
                        d = np.linalg.norm(delta)
                        
                        # Skip same atom (zero distance)
                        if d < 1e-12:
                            continue
                        
                        # Skip if distance exceeds threshold
                        if d > max_d + 1e-6:
                            continue
                        
                        dz = delta[2]
                        # Calculate hopping parameter using Moon-Koshino formula
                        hop = hopping_mk(d, dz)
                        
                        # Skip negligible contributions
                        if abs(hop) < tol:
                            continue
                        
                        # Calculate phase factor
                        k_dot_R = np.dot(k_vec, R_lat)
                        phase = np.exp(1j * k_dot_R)
                        
                        # Accumulate Hamiltonian matrix element
                        H[alpha, beta] += hop * phase
    
    # Compute eigenvalues of Hermitian matrix (sorted in ascending order)
    eigval = np.linalg.eigvalsh(H)
    
    return eigval
