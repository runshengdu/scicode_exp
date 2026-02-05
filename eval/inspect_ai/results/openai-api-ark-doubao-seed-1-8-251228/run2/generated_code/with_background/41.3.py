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
    N = g.shape[0]
    R = g.shape[1]
    
    # Initialize conversion matrix and current abundance (per unit initial abundance)
    M = np.zeros((R, N), dtype=np.float64)
    curr_abundance = np.ones(N, dtype=np.float64)
    
    for k in range(R):
        # Set of depleted resources (1-based) before this niche
        depleted_1 = set(dep_order[:k])
        niche_time = t[k]
        
        for s in range(N):
            # Find the most preferred available resource for species s
            selected_res_1 = None
            for res_1 in pref[s]:
                if res_1 not in depleted_1:
                    selected_res_1 = res_1
                    break
            
            # Convert to 0-based resource index
            r = selected_res_1 - 1
            # Get growth rate for this species-resource pair
            growth_rate = g[s, r]
            
            # Calculate exponential growth factor
            exp_gt = exp(growth_rate * niche_time)
            # Calculate biomass conversion from resource to species during this niche
            conversion = curr_abundance[s] * (exp_gt - 1)
            # Accumulate conversion in the matrix
            M[r, s] += conversion
            
            # Update the species' abundance after this niche
            curr_abundance[s] *= exp_gt
    
    return M


def GetResPts(M):
    '''This function finds the endpoints of the feasibility convex hull
    Inputs:
    M: conversion matrix of biomass, 2d float numpy array of size [R, N]
    Outputs:
    res_pts: a set of points in the resource supply space that marks the region of feasibility. 2d float numpy array of size [R, N].
    '''
    R, N = M.shape
    # Calculate total conversion for each species (sum of each column)
    total_conversion = np.sum(M, axis=0)
    # Normalize each column to sum to 1, forming the resource supply points
    res_pts = M / total_conversion[np.newaxis, :]
    return res_pts



def StrucStability(g, pref, t, dep_order):
    '''This function gets the community's structural stability
    Inputs:
    g: growth rates based on resources, 2d numpy array with dimensions [N, R] and float elements
    pref: species' preference order, 2d numpy array with dimensions [N, R] and int elements between 1 and R
    t: temporal niches, 1d numpy array with length R and float elements
    dep_order: resource depletion order, a tuple of length R with int elements between 1 and R
    Outputs:
    S: structural stability of the community, float
    '''
    # Compute the conversion matrix M using the provided Conversion function
    M = Conversion(g, pref, t, dep_order)
    # Calculate the sum of each column (each sum equals D-1 for balanced state)
    column_sums = M.sum(axis=0)
    # Normalize each column by its sum to form M' where each column sums to 1
    M_prime = M / column_sums[np.newaxis, :]
    # Compute the absolute value of the determinant of M'
    det = np.linalg.det(M_prime)
    S = abs(det)
    return S
