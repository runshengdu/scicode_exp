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


def OneCycle(g, pref, spc_init, Rs, T):
    '''This function simualtes the dynamics in one dilution cycle. 
    Inputs:
    g: growth matrix of species i on resource j. 2d float numpy array of size (N, R). 
    pref: species' preference order, 2d numpy array with dimensions [N, R] and int elements between 1 and R
    spc_init: species abundance at the beginning of cycle. 1d float numpy array of length N. 
    Rs: resource level in environment at the beginning of cycle. 1d float numpy array of length R. 
    T: time span of dilution cycle. float. 
    Outputs: 
    spc_end: species abundance at the end of cycle. 1d float numpy array of length N. 
    Rs_end: resource level in environment at the end of cycle. 1d float numpy array of length R.
    '''
    # Initialize current state
    t_current = 0.0
    N = spc_init.copy()
    R = Rs.copy()
    eps = 1e-12  # Small epsilon for floating point comparisons
    
    while t_current < T - eps:
        # Determine alive species (abundance above threshold)
        alive = (N > eps)
        
        # Get current growth rates and consumed resources
        g_temp, r_temp = SpeciesGrowth(g, pref, R, alive)
        
        # Check if no growth is possible for any alive species
        alive_growth = g_temp[alive]
        if alive_growth.size == 0 or np.all(alive_growth < eps):
            # No more changes, jump to end of cycle
            t_current = T
            break
        
        # Calculate each species' contribution to resource depletion
        contrib = np.where(alive & (r_temp != 0), g_temp * N, 0.0)
        
        # Compute total depletion rate for each resource (0-based)
        C = np.zeros(R.size, dtype=np.float64)
        mask_contrib = (contrib > eps)
        if np.any(mask_contrib):
            # Convert 1-based resource indices to 0-based
            r_0 = r_temp[mask_contrib] - 1
            # Sum contributions per resource
            C += np.bincount(r_0, weights=contrib[mask_contrib], minlength=R.size)
        
        # Calculate time to deplete each resource
        t_depletes = np.full(R.size, np.inf, dtype=np.float64)
        for alpha in range(R.size):
            if R[alpha] > eps and C[alpha] > eps:
                t_depletes[alpha] = R[alpha] / C[alpha]
        min_t_deplete = np.min(t_depletes)
        
        # Determine how much time to advance
        T_remaining = T - t_current
        delta_t = min(min_t_deplete, T_remaining)
        
        # Advance time
        t_current += delta_t
        
        # Update species abundances (exponential growth)
        N *= np.exp(g_temp * delta_t)
        
        # Update resource levels
        R -= C * delta_t
        # Ensure resources don't go negative
        R = np.maximum(R, 0.0)
        
        # Check if all resources are depleted
        if np.all(R < eps):
            t_current = T
            break
    
    # Clean up tiny values to zero for numerical stability
    N[N < eps] = 0.0
    R[R < eps] = 0.0
    
    return N, R



def SimulatedCycles(g, pref, spc_init, Rs, SPC_THRES, T, D, N_cycles):
    '''This function simulates multiple dilution cycles and return the survivors
    Inputs:
    g: growth matrix of species i on resource j. 2d float numpy array of size (N, R). 
    pref: species' preference order, 2d numpy array with dimensions [N, R] and int elements between 1 and R
    spc_init: species abundance at the beginning of cycle. 1d float numpy array of length N. 
    Rs: resource level in environment at the beginning of cycle. 1d float numpy array of length R. 
    SPC_THRES: species dieout cutoff, float
    T: time span of dilution cycle. float. 
    D: dilution rate, float
    N_cycles: number of dilution cycles, int. 
    Outputs: 
    survivors: list of surviving species, elements are integers
    '''
    # Initialize current species and resources with copies to avoid modifying inputs
    current_N = spc_init.copy()
    current_R = Rs.copy()
    
    for _ in range(N_cycles):
        # Simulate one growth phase cycle
        grown_N, grown_R = OneCycle(g, pref, current_N, current_R, T)
        
        # Dilute species abundance by factor D
        diluted_N = grown_N / D
        
        # Mark species as extinct if below threshold
        diluted_N[diluted_N < SPC_THRES] = 0.0
        
        # Dilute remaining resources by factor D
        diluted_R = grown_R / D
        
        # Add fresh resource chunk to diluted resources for next cycle
        current_R = diluted_R + Rs
        
        # Update current species to post-dilution abundances
        current_N = diluted_N
    
    # Collect indices of surviving species (0-based)
    survivors = [idx for idx, abundance in enumerate(current_N) if abundance > 0.0]
    
    return survivors
