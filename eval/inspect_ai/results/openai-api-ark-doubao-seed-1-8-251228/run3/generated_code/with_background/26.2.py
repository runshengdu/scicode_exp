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
    spc_current = spc_init.copy().astype(np.float64)
    Rs_current = Rs.copy().astype(np.float64)
    t_remaining = float(T)
    epsilon = 1e-12
    
    while t_remaining > epsilon and np.any(Rs_current > epsilon):
        # Get current growth rates and resource usage
        g_temp, r_temp_1based = SpeciesGrowth(g, pref, Rs_current, (spc_current > epsilon).astype(bool))
        r_temp_0based = r_temp_1based - 1  # Convert to 0-based index (0→-1 for no resource)
        
        # Collect all possible events
        event_times = [t_remaining]
        event_info = [('time',)]
        
        for alpha in range(Rs_current.size):
            if Rs_current[alpha] <= epsilon:
                continue
            # Find species using this resource
            S_alpha = np.where((spc_current > epsilon) & (r_temp_0based == alpha))[0]
            if len(S_alpha) == 0:
                continue
            # Calculate parameters for resource depletion equation
            C_alpha = np.sum(spc_current[S_alpha])
            D_alpha = C_alpha + Rs_current[alpha]
            g_list = g[S_alpha, alpha]
            
            # Skip if no positive growth rates (resource won't be depleted)
            if not np.any(g_list > epsilon):
                continue
            
            # Define function to solve for depletion time
            def F(t):
                return np.sum(spc_current[S_alpha] * np.exp(g_list * t)) - D_alpha
            
            # Find valid bracket for root finding
            b = 1.0
            max_b = 1e10
            found_bracket = False
            while b < max_b:
                if F(b) > epsilon:
                    found_bracket = True
                    break
                b *= 2
            if not found_bracket:
                continue
            
            # Solve for depletion time
            try:
                sol = root_scalar(F, bracket=[0, b], method='brentq')
                if sol.converged and sol.root > -epsilon:
                    event_times.append(max(sol.root, 0.0))
                    event_info.append(('resource', alpha))
            except Exception:
                continue  # Skip if root finding fails
        
        # Determine earliest event
        min_idx = np.argmin(event_times)
        delta_t = event_times[min_idx]
        current_event = event_info[min_idx]
        
        # Update species abundances
        spc_old = spc_current.copy()
        growth_factors = np.exp(g_temp * delta_t)
        spc_current = spc_old * growth_factors
        # Clamp near-zero abundances to zero
        spc_current[spc_current < epsilon] = 0.0
        
        # Update resource levels
        for alpha in range(Rs_current.size):
            if Rs_current[alpha] <= epsilon:
                continue
            S_alpha = np.where((spc_old > epsilon) & (r_temp_0based == alpha))[0]
            if len(S_alpha) == 0:
                continue
            # Calculate total resource consumption
            consumption = np.sum(spc_current[S_alpha] - spc_old[S_alpha])
            Rs_current[alpha] -= consumption
            # Ensure non-negative resources
            Rs_current[alpha] = max(Rs_current[alpha], 0.0)
        
        # Explicitly set depleted resource to zero if it's the current event
        if current_event[0] == 'resource':
            Rs_current[current_event[1]] = 0.0
        
        # Update remaining time
        t_remaining -= delta_t
        t_remaining = max(t_remaining, 0.0)
    
    # Prepare final outputs
    spc_end = spc_current
    Rs_end = Rs_current
    
    return spc_end, Rs_end
