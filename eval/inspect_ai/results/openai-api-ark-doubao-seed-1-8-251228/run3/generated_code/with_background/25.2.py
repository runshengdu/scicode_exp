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
