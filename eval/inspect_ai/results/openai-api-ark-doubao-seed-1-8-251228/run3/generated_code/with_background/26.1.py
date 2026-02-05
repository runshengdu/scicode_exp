import numpy as np
from math import *
from scipy.optimize import root_scalar
from scipy import special
import copy



def SpeciesGrowth(g, pref, Rs, alive):
    '''This function calcuates the species growth rate
    Inputs:
    g: growth matrix of species i on resource j. 2d float numpy array of size (N, R). 
    pref: species' preference order, 2d numpy array with dimensions [N, R] and int elements between 1 and R
    Rs: resource level in environment. 1d float numpy array of length R. 
    alive: whether the species is present or not. 1d boolean numpy array of length N. 
    Outputs: 
    g_temp: current growth rate of species, 1D float numpy array of length N. 
    r_temp: list of resources that each species is eating. 1D int numpy array of length N. 
    '''
    N = alive.size
    g_temp = np.zeros(N, dtype=np.float64)
    r_temp = np.zeros(N, dtype=np.int32)
    
    for i in range(N):
        if not alive[i]:
            continue
        # Iterate through species' preference order
        for j in range(pref.shape[1]):
            r_pref = pref[i, j]
            r_idx = r_pref - 1  # Convert to 0-based index for Rs and g
            if Rs[r_idx] > 0:
                g_temp[i] = g[i, r_idx]
                r_temp[i] = r_pref
                break
    
    return g_temp, r_temp
