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
