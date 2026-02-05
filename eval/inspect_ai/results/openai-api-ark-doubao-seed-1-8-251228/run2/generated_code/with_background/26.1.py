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
    R = Rs.size
    
    # Initialize output arrays with appropriate data types
    g_temp = np.zeros(N, dtype=np.float64)
    r_temp = np.zeros(N, dtype=np.int32)
    
    # Convert preference order to 0-based indices for numpy array indexing
    pref_0 = pref - 1
    
    # Create matrix indicating if each preferred resource is available (positive level)
    available = Rs[pref_0] > 0.0
    # Determine if each species has any available preferred resource
    any_available = np.any(available, axis=1)
    
    # Find the index of the first available resource in each species' preference order
    first_j = np.argmax(available, axis=1)
    
    # Create mask for species that are alive and have available resources
    mask = alive & any_available
    
    # Update growth rates and consumed resources for eligible species
    if np.any(mask):
        indices = np.where(mask)[0]
        # Get 1-based resource indices for consumed resources
        r_temp[indices] = pref[indices, first_j[indices]]
        # Get corresponding growth rates
        g_temp[indices] = g[indices, pref_0[indices, first_j[indices]]]
    
    return g_temp, r_temp
