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
    # Precompute depletion step for each resource: dep_step[r] is index of r in dep_order
    dep_step = np.zeros(R + 1, dtype=int)  # 1-based resource indexing, index 0 unused
    for k in range(R):
        r = dep_order[k]
        dep_step[r] = k
    
    G = np.zeros_like(g)
    
    for i in range(N):
        pref_i = pref[i]  # Preference list of species i (most to least preferred)
        # Get depletion steps for resources in species i's preference order
        k_list = dep_step[pref_i]
        current_m = 0
        best_r = np.zeros(R, dtype=int)
        
        for j in range(R):
            # Find the first available resource in preference order for niche j
            while current_m < R and k_list[current_m] < j:
                current_m += 1
            best_r[j] = pref_i[current_m]
        
        # Map to growth rates using 0-based column indexing for g
        G[i] = g[i, best_r - 1]
    
    return G
