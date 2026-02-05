from scipy import integrate
from scipy import optimize
import numpy as np



def f_Schrod(energy, l, r_grid):
    '''Input 
    energy: a float
    l: angular momentum quantum number; an int
    r_grid: the radial grid; a 1D array of float
    Output
    f_r: a 1D array of float 
    '''
    Z = 1
    r_B = 0.529  # Bohr radius in Angstroms
    E0 = 13.6    # Rydberg energy in eV
    
    # Convert to dimensionless variables
    x = r_grid / r_B
    eps = energy / E0
    
    # Calculate dimensionless expression from the differential equation
    f_x = l * (l + 1) / (x ** 2) - 2 * Z / x - eps
    
    # Convert to dimensional f(r) in units of 1/Angstrom²
    f_r = f_x / (r_B ** 2)
    
    return f_r
