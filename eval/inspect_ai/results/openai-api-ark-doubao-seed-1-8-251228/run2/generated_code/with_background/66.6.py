import numpy as np
import numpy.linalg as la


def assign_normals(xyzs):
    '''Assign normal vectors on the given atoms
    Args:
        xyzs (np.array): Shape (natoms, 3)
    Returns:
        normed_cross_avg (np.array): Shape (natoms, 3)
    '''
    natoms = xyzs.shape[0]
    normals = np.zeros_like(xyzs)
    
    for i in range(natoms):
        pos = xyzs[i]
        # Compute vectors from current atom to all other atoms
        vectors = xyzs - pos
        # Calculate distances to all other atoms
        distances = la.norm(vectors, axis=1)
        # Get indices of the three nearest neighbors (excluding self)
        sorted_indices = np.argsort(distances)
        neighbor_indices = sorted_indices[1:4]
        neighbor_vectors = vectors[neighbor_indices]
        
        # Compute all three unique cross products between neighbor vectors
        cross1 = np.cross(neighbor_vectors[0], neighbor_vectors[1])
        cross2 = np.cross(neighbor_vectors[0], neighbor_vectors[2])
        cross3 = np.cross(neighbor_vectors[1], neighbor_vectors[2])
        
        # Normalize each cross product
        def normalize_vec(vec):
            norm = la.norm(vec)
            return vec / norm if norm > 1e-12 else np.zeros(3)
        
        norm_cross1 = normalize_vec(cross1)
        norm_cross2 = normalize_vec(cross2)
        norm_cross3 = normalize_vec(cross3)
        
        # Average the normalized cross products and re-normalize
        cross_avg = np.mean([norm_cross1, norm_cross2, norm_cross3], axis=0)
        avg_norm = la.norm(cross_avg)
        if avg_norm < 1e-12:
            # Fallback for degenerate cases (should not occur in graphene)
            initial_normal = np.array([0, 0, -1]) if pos[2] > 0 else np.array([0, 0, 1])
        else:
            initial_normal = cross_avg / avg_norm
        
        # Correct direction based on z-coordinate
        z_coord = pos[2]
        if z_coord > 0:
            # Ensure normal points in negative z-direction
            if initial_normal[2] > 0:
                initial_normal *= -1
        elif z_coord < 0:
            # Ensure normal points in positive z-direction
            if initial_normal[2] < 0:
                initial_normal *= -1
        
        normals[i] = initial_normal
    
    return normals


def potential_repulsive(r_ij, n_i, n_j, z0, C, C0, C2, C4, delta, lamda):
    '''Define repulsive potential.
    Args:
        r_ij: (nmask, 3)
        n_i: (nmask, 3)
        n_j: (nmask, 3)
        z0 (float): KC parameter
        C (float): KC parameter
        C0 (float): KC parameter
        C2 (float): KC parameter
        C4 (float): KC parameter
        delta (float): KC parameter
        lamda (float): KC parameter
    Returns:
        pot (nmask): values of repulsive potential for the given atom pairs.
    '''
    # Compute distance between atom pairs
    r = la.norm(r_ij, axis=1)
    # Exponential term in the repulsive potential
    exp_term = np.exp(-lamda * (r - z0))
    
    # Squared distance for rho calculations
    r_sq = r ** 2
    
    # Dot products of r_ij with normal vectors n_i and n_j
    dot_i = (r_ij * n_i).sum(axis=1)
    dot_j = (r_ij * n_j).sum(axis=1)
    
    # Compute squared rho values and clamp to non-negative to avoid numerical issues
    rho_ij_sq = r_sq - dot_i ** 2
    rho_ij_sq = np.maximum(rho_ij_sq, 0.0)
    
    rho_ji_sq = r_sq - dot_j ** 2
    rho_ji_sq = np.maximum(rho_ji_sq, 0.0)
    
    # Calculate x terms (rho/delta)^2 to avoid square roots
    delta_sq = delta ** 2
    x_ij = rho_ij_sq / delta_sq
    x_ji = rho_ji_sq / delta_sq
    
    # Compute f(rho_ij) and f(rho_ji)
    f_ij = np.exp(-x_ij) * (C0 + C2 * x_ij + C4 * (x_ij ** 2))
    f_ji = np.exp(-x_ji) * (C0 + C2 * x_ji + C4 * (x_ji ** 2))
    
    # Combine terms for the repulsive potential
    bracket_term = C + f_ij + f_ji
    pot = exp_term * bracket_term
    
    return pot


def potential_attractive(rnorm, z0, A):
    '''Define attractive potential.
    Args:
        rnorm (float or np.array): distance
        z0 (float): KC parameter
        A (float): KC parameter
    Returns:
        pot (float): calculated potential
    '''
    pot = -A * (rnorm / z0) ** (-6)
    return pot


def taper(r, rcut):
    '''Define a taper function. This function is 1 at 0 and 0 at rcut.
    Args:
        r (np.array): distance
        rcut (float): always 16 ang    
    Returns:
        result (np.array): taper function values
    '''
    x = r / rcut
    result = np.zeros_like(r)
    # Compute polynomial for x <= 1, set to 0 otherwise
    mask = x <= 1.0
    x_masked = x[mask]
    result[mask] = 20 * x_masked**7 - 70 * x_masked**6 + 84 * x_masked**5 - 35 * x_masked**4 + 1
    return result



def calc_potential(top, bot, z0=3.370060885645178, C0=21.78333851687074, C2=10.469388694543325, C4=8.864962486046355, C=1.3157376477e-05, delta=0.723952360283636, lamda=3.283145920221462, A=13.090159601618883, rcut=16):
    '''Calculate the KC potential energy
    Args:
        top (np.array): (ntop, 3)
        bot (np.array): (nbot, 3)
        z0 (float) : KC parameter
        C0 (float): KC parameter
        C2 (float): KC parameter
        C4 (float): KC parameter
        C (float): KC parameter
        delta (float): KC parameter
        lamda (float): KC parameter
        A (float): KC parameter
        rcut (float): KC parameter
    Returns:
        potential (float): evaluted KC energy
    '''
    # Compute normals for top and bottom layers
    normals_top = assign_normals(top)
    normals_bot = assign_normals(bot)
    
    ntop = top.shape[0]
    nbot = bot.shape[0]
    
    # Compute all pairwise vectors from top to bottom atoms
    r_ij_pairs = bot[np.newaxis, :, :] - top[:, np.newaxis, :]
    r_ij_flat = r_ij_pairs.reshape(-1, 3)
    
    # Compute pairwise distances
    r_flat = la.norm(r_ij_flat, axis=1)
    
    # Compute taper function values
    taper_vals = taper(r_flat, rcut)
    
    # Prepare normal vectors for all pairs
    n_i_flat = np.repeat(normals_top, nbot, axis=0)
    n_j_flat = np.tile(normals_bot, (ntop, 1))
    
    # Compute repulsive potential
    repulsive = potential_repulsive(r_ij_flat, n_i_flat, n_j_flat, z0, C, C0, C2, C4, delta, lamda)
    
    # Compute attractive potential
    attractive = potential_attractive(r_flat, z0, A)
    
    # Compute total V_ij for each pair
    v_ij = repulsive + attractive
    
    # Apply taper function and sum all contributions
    total_potential = np.sum(taper_vals * v_ij)
    
    return total_potential
