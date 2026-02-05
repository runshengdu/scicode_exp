import itertools
import numpy as np
from math import *

def allowed_orders(pref):
    '''Check allowed depletion orders for a set of species with given preference orders
    Input:
    pref: species' preference order, 2d numpy array with dimensions [N, R] and int elements between 1 and R
    Output:
    allowed_orders_list: n_allowed by R, list of tuples with int elements betweem 1 and R. 
    '''
    allowed_orders_list = []
    N, R = pref.shape  # Number of species, number of resources
    all_permutations = itertools.permutations(range(1, R + 1))
    
    for perm in all_permutations:
        available = set(range(1, R + 1))
        valid = True
        for o in perm:
            if o not in available:
                valid = False
                break
            
            has_consumer = False
            for s_pref in pref:
                # Find the index of current resource in the species' preference list
                idx = np.where(s_pref == o)[0][0]
                # Resources that the species prefers over current resource
                preferred_over = s_pref[:idx]
                # Check if any preferred resource is still available
                any_preferred_available = np.any(np.isin(preferred_over, list(available)))
                if not any_preferred_available:
                    has_consumer = True
                    break  # No need to check other species
            
            if not has_consumer:
                valid = False
                break
            
            available.remove(o)
        
        if valid:
            allowed_orders_list.append(perm)
    
    return allowed_orders_list



def G_mat(g, pref, dep_order):
    '''Convert to growth rates based on temporal niches
    Input
    g: growth rates based on resources, 2d numpy array with dimensions [N, R] and float elements
    pref: species' preference order, 2d numpy array with dimensions [N, R] and int elements between 1 and R
    dep_order: resource depletion order, a tuple of length R with int elements between 1 and R
    Output
    G: "converted" growth rates based on temporal niches, 2d numpy array with dimensions [N, R]
    '''
    N, R = g.shape
    G = np.zeros((N, R), dtype=float)
    
    for j in range(R):
        # Resources depleted before the j-th temporal niche
        depleted = set(dep_order[:j])
        # Available resources at the start of the j-th niche
        available = set(range(1, R + 1)) - depleted
        available_list = list(available)
        
        # Create mask indicating which resources in preference lists are available
        mask = np.isin(pref, available_list)
        # Find index of first available resource for each species
        first_available_idx = np.argmax(mask, axis=1)
        # Get the chosen resource IDs for all species
        chosen_resources = pref[np.arange(N), first_available_idx]
        # Map to growth rates using 0-based column indexing for g
        G[:, j] = g[np.arange(N), chosen_resources - 1]
    
    return G



def check_G_feasibility(G, D):
    N, R = G.shape
    # If there are more resources than species, full column rank is impossible
    if R > N:
        return False
    
    # Calculate rank of the growth matrix
    rank_G = np.linalg.matrix_rank(G)
    if rank_G < R:
        return False
    
    # Prepare the target vector of ones
    ones_vec = np.ones(N)
    # Solve the linear system using least squares
    t0, residuals, _, _ = np.linalg.lstsq(G, ones_vec, rcond=None)
    
    # Check if the system is consistent (residuals are negligible)
    if residuals.size > 0 and residuals[0] > 1e-9:
        return False
    
    # Check if all elements of the solution vector are positive (allowing minor numerical errors)
    if np.all(t0 > 1e-12):
        return True
    else:
        return False
