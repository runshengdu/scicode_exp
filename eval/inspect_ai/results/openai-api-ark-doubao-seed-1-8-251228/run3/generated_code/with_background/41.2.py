import numpy as np
from math import exp



def Conversion(g, pref, t, dep_order):
    '''This function calculates the biomass conversion matrix M
    Inputs:
    g: growth rates based on resources, 2d numpy array with dimensions [N, R] and float elements
    pref: species' preference order, 2d numpy array with dimensions [N, R] and int elements between 1 and R
    t: temporal niches, 1d numpy array with length R and float elements
    dep_order: resource depletion order, a tuple of length R with int elements between 1 and R
    Outputs:
    M: conversion matrix of biomass from resource to species. 2d float numpy array with dimensions [R, N].
    '''
    N, R = g.shape
    # Initialize conversion matrix M with zeros
    M = np.zeros((R, N), dtype=float)
    
    # Precompute depletion_k: maps 1-based resource index to the niche when it's depleted
    depletion_k = np.zeros(R + 1, dtype=int)  # Index 0 unused (resources are 1-based)
    for k in range(R):
        resource = dep_order[k]
        depletion_k[resource] = k
    
    for species in range(N):
        # Get preference list for current species (1-based resource indices)
        species_prefs = pref[species]
        # Get depletion niches for resources in preference order
        depletion_times = depletion_k[species_prefs]
        
        # Use two-pointer technique to find best resource for each niche
        best_resource_indices = np.zeros(R, dtype=int)
        current_pref_idx = 0
        for niche in range(R):
            # Move to next preferred resource if current is unavailable
            while current_pref_idx < R and depletion_times[current_pref_idx] < niche:
                current_pref_idx += 1
            best_resource_indices[niche] = current_pref_idx
        
        # Simulate growth and calculate conversions
        current_abundance = 1.0
        for niche in range(R):
            pref_idx = best_resource_indices[niche]
            resource_1based = species_prefs[pref_idx]
            resource_code = resource_1based - 1  # Convert to 0-based index
            
            growth_rate = g[species, resource_code]
            niche_duration = t[niche]
            exp_term = exp(growth_rate * niche_duration)
            
            # Calculate biomass converted from resource to species
            conversion = current_abundance * (exp_term - 1)
            M[resource_code, species] += conversion
            
            # Update species abundance for next niche
            current_abundance *= exp_term
    
    return M



def GetResPts(M):
    '''This function finds the endpoints of the feasibility convex hull
    Inputs:
    M: conversion matrix of biomass, 2d float numpy array of size [R, N]
    Outputs:
    res_pts: a set of points in the resource supply space that marks the region of feasibility. 2d float numpy array of size [R, N].
    '''
    # Calculate sum of each column (total resource conversion per species)
    column_sums = M.sum(axis=0)
    # Normalize each column by its sum to get points on the simplex (sum R_i = 1)
    res_pts = M / column_sums[np.newaxis, :]
    return res_pts
