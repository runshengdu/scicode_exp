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
        thz_to_ev = 0.004135667
        k_boltz_ev = 8.617333262e-5  # Boltzmann constant in eV per Kelvin
        exponent = (freq * thz_to_ev) / (k_boltz_ev * temp)
        nbose = 1 / (np.exp(exponent) - 1)
    return nbose
