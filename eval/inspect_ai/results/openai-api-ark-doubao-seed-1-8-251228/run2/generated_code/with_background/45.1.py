import numpy as np



def init_grid(Nt, Nx, Ny, x_split, T1, alpha1, T2, alpha2):
    '''Initialize a 3d array for storing the temperature values. The array has one time dimension and two real space dimensions.
    Initialize a 2d array for storing the thermal diffusivity corresponding to each grid point in real space.  
    There could be a vertical boundary in real space that represents the interface between two different materials.
    Input
    Nt: time dimension of the temperature grid; int
    Nx: x-dimension (number of columns) of the temperature grid; int
    Ny: y-dimension (number of rows) of the temperature grid; int
    x_split: the column index of the vertical interface. All columns up to and including this index (material 1) will have T1 and alpha1, and all columns with a larger index (material 2) will have T2 and alpha2; int
    T1: the initial temperature of each grid point for material 1(in Celsius); float
    alpha1: the thermal diffusivity of material 1; float
    T2: the initial temperature of each grid point for material 2(in Celsius); float
    alpha2: the heat conductivity of material 2; float
    Output
    temp_grid: temperature grid; 3d array of floats
    diff_grid: thermal diffusivity grid; 2d array of floats
    '''
    # Initialize temperature grid with zeros for all time steps
    temp_grid = np.zeros((Nt, Ny, Nx), dtype=np.float64)
    
    # Fill initial temperature for first time step
    # Material 1: columns 0 to x_split (inclusive)
    temp_grid[0, :, :x_split + 1] = T1
    # Material 2: columns x_split+1 to Nx-1
    temp_grid[0, :, x_split + 1:] = T2
    
    # Initialize diffusivity grid with material 2 values first
    diff_grid = np.full((Ny, Nx), alpha2, dtype=np.float64)
    # Set material 1 values for columns up to x_split
    diff_grid[:, :x_split + 1] = alpha1
    
    return temp_grid, diff_grid
