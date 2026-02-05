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
