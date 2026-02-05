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
    # Calculate resource value term (element-wise product of efficiency and level)
    resource_value = w * res
    # Compute total resource contribution for each species
    total_contribution = c @ resource_value
    # Calculate net gain after maintenance cost
    net_gain = total_contribution - m
    # Compute final growth rate by scaling with inverse timescale
    g_spc = b * net_gain
    return g_spc
