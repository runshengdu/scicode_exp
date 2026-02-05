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



def check_G_feasibility(G, D):
    '''Determine if a "converted" growth rate matrix G leads to a feasible coexistence. 
    Input 
    G: growth rate based on temporal niches, 2d numpy float array with dimensions [N, R]
    D: dilution factor, float
    Output
    feasible: boolean
    '''
    N, R = G.shape
    
    # Handle invalid dilution factor values
    if D <= 0:
        return False
    if D == 1.0:
        # System reduces to Gt = 0, no positive unique solution exists
        return False
    
    # Create target vector where each element is ln(D)
    b = np.full(N, np.log(D))
    
    # Check consistency and uniqueness of linear system solution
    rank_G = np.linalg.matrix_rank(G)
    augmented_matrix = np.hstack((G, b.reshape(-1, 1)))
    rank_augmented = np.linalg.matrix_rank(augmented_matrix)
    
    # System has no solution or infinitely many solutions
    if rank_G != rank_augmented or rank_G != R:
        return False
    
    # Solve the linear system
    t, _, _, _ = np.linalg.lstsq(G, b, rcond=None)
    
    # Verify solution accuracy (account for numerical precision)
    computed_b = G @ t
    if not np.allclose(computed_b, b, atol=1e-9):
        return False
    
    # Check if all temporal niche lengths are positive (account for numerical precision)
    if np.all(t > -1e-9) and np.min(t) > -1e-9:
        # Ensure no significant negative values and all are effectively positive
        return np.all(t > -1e-9) and np.sum(t > 0) == R
    else:
        return False
