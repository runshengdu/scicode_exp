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
    N, R = pref.shape
    # Precompute position of each resource in each species' preference list
    pos = np.zeros((N, R + 1), dtype=int)  # pos[n][r] is index of resource r in species n's list
    for n in range(N):
        row = pref[n]
        for idx in range(R):
            r = row[idx]
            pos[n][r] = idx
    
    # Precompute constraints: list of (s, r) where s must come before r
    constraints = []
    for s in range(1, R + 1):
        for r in range(1, R + 1):
            if s == r:
                continue
            # Check if all species prefer s over r
            if np.all(pos[:, s] < pos[:, r]):
                constraints.append((s, r))
    
    allowed_orders_list = []
    # Generate all possible permutations of resources
    for perm in itertools.permutations(range(1, R + 1)):
        # Create position map for current permutation
        perm_pos = {res: idx for idx, res in enumerate(perm)}
        valid = True
        for s, r in constraints:
            if perm_pos[s] >= perm_pos[r]:
                valid = False
                break
        if valid:
            allowed_orders_list.append(perm)
    
    return allowed_orders_list
