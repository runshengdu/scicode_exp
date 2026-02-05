import numpy as np
import itertools

def ground_state_wavelength(L, mr):
    '''Given the width of a infinite square well, provide the corresponding wavelength of the ground state eigen-state energy.
    Input:
    L (float): Width of the infinite square well (nm).
    mr (float): relative effective electron mass.
    Output:
    lmbd (float): Wavelength of the ground state energy (nm).
    '''
    # Physical constants in SI units
    m0 = 9.109e-31  # Free electron mass (kg)
    c = 3e8         # Speed of light (m/s)
    h = 6.626e-34   # Planck constant (J·s)
    
    # Convert well width from nanometers to meters
    L_m = L * 1e-9
    
    # Calculate effective electron mass
    m_eff = mr * m0
    
    # Calculate photon wavelength in meters using simplified formula
    lambda_m = (8 * c * m_eff * L_m ** 2) / h
    
    # Convert wavelength from meters to nanometers
    lmbd = lambda_m * 1e9
    
    return lmbd



def generate_quadratic_combinations(x, y, z, N):
    '''With three numbers given, return an array with the size N that contains the smallest N numbers which are quadratic combinations of the input numbers.
    Input:
    x (float): The first number.
    y (float): The second number.
    z (float): The third number.
    Output:
    C (size N numpy array): The collection of the quadratic combinations.
    '''
    # Generate all possible (i, j, k) tuples where i, j, k are positive integers up to N
    indices = itertools.product(range(1, N + 1), repeat=3)
    
    # Calculate all quadratic combination sums
    sums = []
    for i, j, k in indices:
        current_sum = i**2 * x + j**2 * y + k**2 * z
        sums.append(current_sum)
    
    # Sort the sums in ascending order
    sums.sort()
    
    # Select the first N smallest sums and convert to numpy array
    result = np.array(sums[:N])
    
    return result
