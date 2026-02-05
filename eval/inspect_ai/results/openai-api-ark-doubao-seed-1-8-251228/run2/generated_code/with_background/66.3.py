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
