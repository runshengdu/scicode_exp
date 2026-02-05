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
