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
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    # Calculate coordinates for sublattice A
    x_A = 0.5 * a * (i_flat - j_flat)
    y_A = (a * np.sqrt(3) / 2) * (i_flat + j_flat) + s
    z_A = np.full_like(x_A, z)
    
    # Calculate coordinates for sublattice B
    x_B = x_A
    y_B = (a * np.sqrt(3) / 2) * (i_flat + j_flat) + (a * np.sqrt(3) / 3) + s
    z_B = np.full_like(x_B, z)
    
    # Combine both sublattices into a single array
    atoms_A = np.column_stack((x_A, y_A, z_A))
    atoms_B = np.column_stack((x_B, y_B, z_B))
    atoms = np.vstack((atoms_A, atoms_B))
    
    return atoms


def assign_normals(xyzs):
    '''Assign normal vectors on the given atoms
    Args:
        xyzs (np.array): Shape (natoms, 3)
    Returns:
        normed_cross_avg (np.array): Shape (natoms, 3)
    '''
    natoms = xyzs.shape[0]
    normals = np.zeros((natoms, 3), dtype=np.float64)

    for i in range(natoms):
        pos_i = xyzs[i]
        # Compute vectors from pos_i to all other atoms
        vectors = xyzs - pos_i  # shape (natoms, 3)
        # Compute squared distances to avoid sqrt
        dist_sq = np.sum(vectors ** 2, axis=1)
        # Sort indices by distance squared
        sorted_indices = np.argsort(dist_sq)
        # Get indices of three nearest neighbors (skip self)
        neighbor_indices = sorted_indices[1:4]
        # Get the three neighbor vectors
        vecs = vectors[neighbor_indices]  # shape (3, 3)

        # Compute all three pairwise cross products
        cross1 = np.cross(vecs[0], vecs[1])
        cross2 = np.cross(vecs[0], vecs[2])
        cross3 = np.cross(vecs[1], vecs[2])

        # Normalize each cross product
        norm1 = la.norm(cross1)
        normed_cross1 = cross1 / norm1 if norm1 >= 1e-16 else np.zeros_like(cross1)
        
        norm2 = la.norm(cross2)
        normed_cross2 = cross2 / norm2 if norm2 >= 1e-16 else np.zeros_like(cross2)
        
        norm3 = la.norm(cross3)
        normed_cross3 = cross3 / norm3 if norm3 >= 1e-16 else np.zeros_like(cross3)

        # Average the normalized cross products
        avg_cross = (normed_cross1 + normed_cross2 + normed_cross3) / 3.0

        # Normalize the average to get unit normal
        avg_norm = la.norm(avg_cross)
        if avg_norm < 1e-16:
            # Handle degenerate case with default direction
            normed_avg = np.array([0.0, 0.0, 1.0])
        else:
            normed_avg = avg_cross / avg_norm

        # Correct direction based on atom's z coordinate
        atom_z = pos_i[2]
        if atom_z > 0:
            # Flip if normal points positive z
            if normed_avg[2] > 0:
                normed_avg *= -1
        elif atom_z < 0:
            # Flip if normal points negative z
            if normed_avg[2] < 0:
                normed_avg *= -1

        # Assign to normals array
        normals[i] = normed_avg

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
    # Compute magnitude of r_ij vectors
    r = la.norm(r_ij, axis=1)
    r_sq = r ** 2
    
    # Calculate dot products between r_ij and normal vectors
    dot_ni = np.sum(r_ij * n_i, axis=1)
    dot_ni_sq = dot_ni ** 2
    dot_nj = np.sum(r_ij * n_j, axis=1)
    dot_nj_sq = dot_nj ** 2
    
    # Compute squared transverse distances (clamp to non-negative to avoid numerical issues)
    rho_ij_sq = np.maximum(r_sq - dot_ni_sq, 0.0)
    rho_ji_sq = np.maximum(r_sq - dot_nj_sq, 0.0)
    
    # Calculate (rho/delta)^2 terms for f(rho) function
    delta_sq = delta ** 2
    x_ij = rho_ij_sq / delta_sq
    x_ji = rho_ji_sq / delta_sq
    
    # Evaluate f(rho) for both transverse distances
    f_ij = np.exp(-x_ij) * (C0 + C2 * x_ij + C4 * (x_ij ** 2))
    f_ji = np.exp(-x_ji) * (C0 + C2 * x_ji + C4 * (x_ji ** 2))
    
    # Compute the bracket term in the potential formula
    bracket_term = C + f_ij + f_ji
    
    # Compute the exponential prefactor
    exp_term = np.exp(-lamda * (r - z0))
    
    # Calculate final repulsive potential values
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
        result (np.array): taper functioin values
    '''
    x = r / rcut
    # Compute the polynomial for x <= 1
    polynomial = 20 * x**7 - 70 * x**6 + 84 * x**5 - 35 * x**4 + 1
    # Set to 0 where x > 1, keep polynomial value otherwise
    result = np.where(x <= 1.0, polynomial, 0.0)
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
    # Compute normal vectors for top and bottom layers
    normals_top = assign_normals(top)
    normals_bot = assign_normals(bot)
    
    # Generate all pairwise distance vectors between top and bottom atoms
    top_reshaped = top[:, np.newaxis, :]
    bot_reshaped = bot[np.newaxis, :, :]
    r_ij = bot_reshaped - top_reshaped  # Vector from top atom i to bottom atom j
    r_ij_flat = r_ij.reshape(-1, 3)
    
    # Prepare normal vectors for all atom pairs
    normals_top_reshaped = normals_top[:, np.newaxis, :]
    normals_bot_reshaped = normals_bot[np.newaxis, :, :]
    n_i_flat = normals_top_reshaped.reshape(-1, 3)
    n_j_flat = normals_bot_reshaped.reshape(-1, 3)
    
    # Calculate distance magnitudes
    r_norm = la.norm(r_ij_flat, axis=1)
    
    # Compute repulsive and attractive potential components
    repulsive = potential_repulsive(r_ij_flat, n_i_flat, n_j_flat, z0, C, C0, C2, C4, delta, lamda)
    attractive = potential_attractive(r_norm, z0, A)
    
    # Total potential per pair
    v_ij = repulsive + attractive
    
    # Apply taper function
    tap = taper(r_norm, rcut)
    tapered_v = v_ij * tap
    
    # Sum all contributions to get total potential energy
    total_potential = np.sum(tapered_v)
    
    return total_potential
