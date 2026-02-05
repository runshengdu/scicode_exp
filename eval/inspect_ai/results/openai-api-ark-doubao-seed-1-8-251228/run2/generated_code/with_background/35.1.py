import numpy as np
import itertools



def ground_state_wavelength(L, mr):
    '''Given the width of a infinite square well, provide the corresponding wavelength of the ground state eigen-state energy.
    Input:
    L (float): Width of the infinite square well (nm).
    mr (float): relative effective electron mass.
    Output:
    lmbd (float): Wavelength of the ground state energy (nm).
    '''
    m0 = 9.109e-31
    c = 3e8
    h = 6.626e-34
    lmbd = (8 * c * mr * m0 * (L ** 2) * 1e-9) / h
    return lmbd
