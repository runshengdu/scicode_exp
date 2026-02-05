import numpy as np
import numpy.linalg as la

def generate_monolayer_graphene(s, a, z, n):
    '''Generate the geometry of monolayer graphene.
    Args:
        s (float): Horizontal in-plane sliding distance.
        a (float): Lattice constant.
        z (float): z-coordinate
        n (int): supercell size
    Returns:
        atoms (np.array): Array containing the x, y, and z coordinates of the atoms.
    '''
    # Generate grid of indices for supercell expansion
    i = np.arange(-n, n + 1)
    j = np.arange(-n, n + 1)
    i_grid, j_grid = np.meshgrid(i, j, indexing='ij')
    
    # Calculate coordinates for sublattice A
    x_a = (a * np.sqrt(3) / 2) * (i_grid + j_grid)
    y_a = (a / 2) * (j_grid - i_grid) + s
    z_a = np.full_like(x_a, z)
    
    # Calculate coordinates for sublattice B (shifted in x-direction relative to A)
    x_b = x_a + (a * np.sqrt(3) / 3)
    y_b = y_a
    z_b = z_a
    
    # Reshape and concatenate both sublattices into a single array
    atoms_a = np.stack([x_a.ravel(), y_a.ravel(), z_a.ravel()], axis=1)
    atoms_b = np.stack([x_b.ravel(), y_b.ravel(), z_b.ravel()], axis=1)
    atoms = np.concatenate([atoms_a, atoms_b], axis=0)
    
    return atoms


def assign_normals(xyzs):
    '''Assign normal vectors on the given atoms
    Args:
        xyzs (np.array): Shape (natoms, 3)
    Returns:
        normed_cross_avg (np.array): Shape (natoms, 3)
    '''
    natoms = xyzs.shape[0]
    normals = np.zeros((natoms, 3), dtype=xyzs.dtype)
    eps = 1e-12  # Small tolerance for floating point comparisons
    
    for i in range(natoms):
        # Efficiently compute squared distances to all other atoms
        xyz_i = xyzs[i]
        dist_sq = np.sum(xyzs**2, axis=1) + np.sum(xyz_i**2) - 2 * xyzs.dot(xyz_i)
        
        # Get indices of three nearest neighbors (excluding self)
        sorted_indices = np.argsort(dist_sq)
        nn_indices = sorted_indices[1:4]
        
        # Calculate vectors from current atom to its nearest neighbors
        v1 = xyzs[nn_indices[0]] - xyz_i
        v2 = xyzs[nn_indices[1]] - xyz_i
        v3 = xyzs[nn_indices[2]] - xyz_i
        
        # Compute cross products for all unordered pairs of neighbor vectors
        cp1 = np.cross(v1, v2)
        cp2 = np.cross(v1, v3)
        cp3 = np.cross(v2, v3)
        
        # Normalize each cross product to unit vectors
        def normalize_vector(vec):
            vec_norm = la.norm(vec)
            return vec / vec_norm if vec_norm > eps else np.array([0.0, 0.0, 1.0])
        
        u1 = normalize_vector(cp1)
        u2 = normalize_vector(cp2)
        u3 = normalize_vector(cp3)
        
        # Average the three normalized cross products
        avg_u = (u1 + u2 + u3) / 3
        
        # Normalize the average to get initial normal vector
        avg_norm = la.norm(avg_u)
        initial_normal = avg_u / avg_norm if avg_norm > eps else np.array([0.0, 0.0, 1.0])
        
        # Correct direction based on atom's z-coordinate
        atom_z = xyz_i[2]
        if atom_z > eps:
            # Ensure normal points in negative z-direction
            if initial_normal[2] > 0:
                initial_normal *= -1
        elif atom_z < -eps:
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
    # Compute scalar distance for each pair
    r = la.norm(r_ij, axis=1)
    r_sq = r ** 2
    
    # Compute dot products of r_ij with normals n_i and n_j
    dot_ni = np.sum(r_ij * n_i, axis=1)
    dot_nj = np.sum(r_ij * n_j, axis=1)
    
    # Compute squared transverse distances rho_ij and rho_ji (clamped to non-negative)
    rho_ij_sq = np.maximum(r_sq - dot_ni ** 2, 0.0)
    rho_ji_sq = np.maximum(r_sq - dot_nj ** 2, 0.0)
    
    # Compute t = (rho/delta)^2 for both transverse distances
    delta_sq = delta ** 2
    t_ij = rho_ij_sq / delta_sq
    t_ji = rho_ji_sq / delta_sq
    
    # Calculate f(rho_ij) and f(rho_ji) using the polynomial form
    f_ij = np.exp(-t_ij) * (C0 + C2 * t_ij + C4 * (t_ij ** 2))
    f_ji = np.exp(-t_ji) * (C0 + C2 * t_ji + C4 * (t_ji ** 2))
    
    # Compute the bracket term in the potential formula
    bracket = C + f_ij + f_ji
    
    # Calculate the exponential decay factor
    exp_factor = np.exp(-lamda * (r - z0))
    
    # Compute final repulsive potential values
    pot = exp_factor * bracket
    
    return pot
