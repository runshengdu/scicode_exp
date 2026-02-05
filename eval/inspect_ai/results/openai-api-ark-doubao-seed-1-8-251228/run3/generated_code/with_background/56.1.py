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
