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
