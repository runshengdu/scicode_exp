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
    # Calculate V_ppπ using the exponential decay formula
    v_ppπ = v_p0 * np.exp(-b * (d - a0))
    # Calculate V_ppσ using the exponential decay formula
    v_ppσ = v_s0 * np.exp(-b * (d - d0))
    # Compute the squared ratio of out-of-plane distance to total distance
    dz_over_d_sq = (dz / d) ** 2
    # Calculate the hopping parameter using the weighted combination of V_ppπ and V_ppσ
    hopping = v_ppπ * (1 - dz_over_d_sq) + v_ppσ * dz_over_d_sq
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
    # Compute displacement vectors for the unit cells
    disp_i = di @ latvecs
    disp_j = dj @ latvecs
    
    # Get basis positions for the specified atoms
    basis_i = basis[ai]
    basis_j = basis[aj]
    
    # Calculate absolute positions of the atoms
    pos_i = basis_i + disp_i
    pos_j = basis_j + disp_j
    
    # Compute vector between the two atoms
    delta = pos_i - pos_j
    
    # Calculate distance between atoms and out-of-plane component
    d = np.linalg.norm(delta, axis=1)
    dz = delta[:, 2]
    
    # Evaluate hopping using Moon-Koshino parameterization
    hopping = hopping_mk(d, dz)
    
    return hopping
