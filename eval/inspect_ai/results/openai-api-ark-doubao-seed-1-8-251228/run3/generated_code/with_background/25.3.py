import numpy as np
from scipy.integrate import solve_ivp
from functools import partial

def SpeciesGrowth(spc, res, b, c, w, m):
    '''This function calcuates the species growth rate
    Inputs:
    res: resource level, 1D array of length R
    b: inverse timescale of species dynamics, 1D array of length N
    c: consumer-resource conversion matrix, 2D array of shape [N, R]
    w: value/efficiency of resoruce, 1D array of length R
    m: species maintainance cost, 1D array of length N
    Outputs: 
    g_spc: growth rate of species, 1D array of length N
    '''
    # Calculate the weighted resource contribution for each species
    weighted_resource = c @ (w * res)
    # Compute net gain after subtracting maintenance cost
    net_gain = weighted_resource - m
    # Calculate growth rate using inverse timescale
    g_spc = b * net_gain
    
    return g_spc


def ResourcesUpdate(spc, res, c, r, K):
    '''This function calculates the changing rates of resources
    Inputs:
    spc: species population, 1D array of length N
    res: resource abundance, 1D array of length R
    c: consumer-resource conversion matrix, 2D array of shape [N, R]
    r: inverse timescale of resource growth, 1D array of length R
    K: resource carrying capacity, 1D array of length R
    Outputs: 
    f_res: growth rate of resources, 1D array of length R
    '''
    # Calculate logistic growth component for each resource
    logistic_component = r * (1 - res / K)
    # Calculate total resource consumption by all species
    total_consumption = spc @ c
    # Compute net resource growth factor
    net_growth_factor = logistic_component - total_consumption
    # Calculate resource dynamics using the given formula
    f_res = res * net_growth_factor
    
    return f_res



def Simulate(spc_init, res_init, b, c, w, m, r, K, tf, dt, SPC_THRES):
    '''This function simulates the model's dynamics
    Inputs:
    spc_init: initial species population, 1D array of length N
    res_init: initial resource abundance, 1D array of length R
    b: inverse timescale of species dynamics, 1D array of length N
    c: consumer-resource conversion matrix, 2D array of shape [N, R]
    w: value/efficiency of resoruce, 1D array of length R
    m: species maintainance cost, 1D array of length N
    r: inverse timescale of resource growth, 1D array of length R
    K: resource carrying capacity, 1D array of length R
    tf: final time, float
    dt: timestep length, float
    SPC_THRES: species dieout cutoff, float
    Outputs: 
    survivors: list of integers (values between 0 and N-1)
    '''
    # Initialize current state with proper data type
    current_spc = np.array(spc_init, dtype=np.float64).copy()
    current_res = np.array(res_init, dtype=np.float64).copy()
    
    # Initialize alive mask: species are alive if above threshold initially
    alive = current_spc > SPC_THRES
    # Set initial species below threshold to zero
    current_spc[~alive] = 0.0
    
    t = 0.0
    
    while t < tf:
        # Calculate species growth rate, set zero for extinct species
        g_spc = SpeciesGrowth(current_spc, current_res, b, c, w, m)
        g_spc[~alive] = 0.0
        
        # Calculate resource dynamics
        f_res = ResourcesUpdate(current_spc, current_res, c, r, K)
        
        # Ensure we don't overshoot the final time
        step = min(dt, tf - t)
        
        # Update species and resources with Euler step
        next_spc = current_spc + g_spc * step
        next_res = current_res + f_res * step
        
        # Enforce non-negative species abundances
        next_spc = np.maximum(next_spc, 0.0)
        
        # Update alive status: only retain alive if still above threshold
        alive = alive & (next_spc > SPC_THRES)
        
        # Set extinct species to zero to prevent resource consumption
        next_spc[~alive] = 0.0
        
        # Enforce non-negative resource levels
        next_res = np.maximum(next_res, 0.0)
        
        # Update state and time for next iteration
        current_spc, current_res = next_spc, next_res
        t += step
    
    # Collect indices of surviving species
    survivors = np.where(alive)[0].tolist()
    
    return survivors
