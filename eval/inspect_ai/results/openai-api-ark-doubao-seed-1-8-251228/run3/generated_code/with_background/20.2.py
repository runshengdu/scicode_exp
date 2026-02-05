import numpy as np

def bose_distribution(freq, temp):
    '''This function defines the bose-einstein distribution
    Input
    freq: a 2D numpy array of dimension (nqpts, nbnds) that contains the phonon frequencies; each element is a float. For example, freq[0][1] is the phonon frequency of the 0th q point on the 1st band
    temp: a float representing the temperature of the distribution
    Output
    nbose: A 2D array of the same shape as freq, representing the Bose-Einstein distribution factor for each frequency.
    '''
    if temp == 0.0:
        nbose = np.zeros_like(freq)
    else:
        conversion_factor = 0.004135667
        k_boltzmann = 8.617333262e-5  # eV/K
        exponent = (freq * conversion_factor) / (temp * k_boltzmann)
        exp_term = np.exp(exponent) - 1.0
        nbose = 1.0 / exp_term
    return nbose




def phonon_angular_momentum(freq, polar_vec, temp):
    '''    Calculate the phonon angular momentum based on predefined axis orders: alpha=z, beta=x, gamma=y.
        Input
        freq: a 2D numpy array of dimension (nqpts, nbnds) that contains the phonon frequencies; each element is a float. For example, freq[0][1] is the phonon frequency of the 0th q point on the 1st band
        polar_vec: a numpy array of shape (nqpts, nbnds, natoms, 3) that contains the phonon polarization vectors; each element is a numpy array of 3 complex numbers. 
        nqpts is the number of k points. nbnds is the number of bands. natoms is the number of atoms. For example, polar_vec[0][1][2][:] represents the 1D array of x,y,z components of the 
        polarization vector of the 0th q point of the 1st band of the 2nd atom.
        temp: a float representing the temperature of the distribution in Kelvin
        Output
        momentum: A 3D array containing the mode decomposed phonon angular momentum. The dimension is (3, nqpts, nbnds). For example, momentum[0][1][2] is the x-component
    of the phonon angular momentum of the 1st q point on the 2nd band
        Notes:
        - Angular momentum values are in units of ħ (reduced Planck constant).
        
    '''
    # Calculate Bose-Einstein distribution
    n0 = bose_distribution(freq, temp)
    occ_factor = n0 + 0.5  # Occupation factor including zero-point energy
    
    # Calculate l^x component: 2 * Im(sum_j ε_j^y* * ε_j^z)
    conj_y = np.conj(polar_vec[..., 1])
    z_comp = polar_vec[..., 2]
    product_y_z = conj_y * z_comp
    sum_y_z = np.sum(product_y_z, axis=2)  # Sum over atoms
    l_x = 2 * np.imag(sum_y_z)
    
    # Calculate l^y component: 2 * Im(sum_j ε_j^z* * ε_j^x)
    conj_z = np.conj(polar_vec[..., 2])
    x_comp = polar_vec[..., 0]
    product_z_x = conj_z * x_comp
    sum_z_x = np.sum(product_z_x, axis=2)  # Sum over atoms
    l_y = 2 * np.imag(sum_z_x)
    
    # Calculate l^z component: 2 * Im(sum_j ε_j^x* * ε_j^y)
    conj_x = np.conj(polar_vec[..., 0])
    y_comp = polar_vec[..., 1]
    product_x_y = conj_x * y_comp
    sum_x_y = np.sum(product_x_y, axis=2)  # Sum over atoms
    l_z = 2 * np.imag(sum_x_y)
    
    # Compute mode-decomposed angular momentum
    mode_l_x = occ_factor * l_x
    mode_l_y = occ_factor * l_y
    mode_l_z = occ_factor * l_z
    
    # Stack components into output shape (3, nqpts, nbnds)
    momentum = np.stack([mode_l_x, mode_l_y, mode_l_z], axis=0)
    
    return momentum
