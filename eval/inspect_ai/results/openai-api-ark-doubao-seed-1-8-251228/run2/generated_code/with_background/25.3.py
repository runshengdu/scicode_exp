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
    logistic_growth = r * (1 - res / K)
    # Compute total consumption of each resource by all species
    total_consumption = spc @ c
    # Calculate net rate before scaling with current resource level
    net_rate = logistic_growth - total_consumption
    # Compute final resource dynamics
    f_res = res * net_rate
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
    N = len(spc_init)
    R = len(res_init)
    
    # Convert initial states to numpy arrays for numerical operations
    spc_init_arr = np.asarray(spc_init, dtype=float)
    res_init_arr = np.asarray(res_init, dtype=float)
    
    # Define the combined ODE function for species and resources
    def dy_dt(t, y):
        spc = y[:N]
        res = y[N:]
        # Compute species growth rates using the provided function
        g_spc = SpeciesGrowth(spc, res, b, c, w, m)
        # Compute resource dynamics using the provided function
        f_res = ResourcesUpdate(spc, res, c, r, K)
        # Combine derivatives into a single array for solve_ivp
        return np.concatenate([g_spc, f_res])
    
    # Initial state vector combining species and resources
    y0 = np.concatenate([spc_init_arr, res_init_arr])
    
    # Integrate the ODE system from t=0 to t=tf
    # Use max_step=dt to ensure integration steps do not exceed the specified timestep
    sol = solve_ivp(dy_dt, t_span=(0, tf), y0=y0, max_step=dt)
    
    # Extract final species abundances from the solution
    final_spc = sol.y[:N, -1]
    
    # Enforce non-negative species abundances (physical constraint)
    final_spc = np.maximum(final_spc, 0.0)
    
    # Identify species with abundance above the dieout threshold
    survivors = np.where(final_spc > SPC_THRES)[0].tolist()
    
    return survivors
