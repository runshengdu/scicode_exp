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



def absorption(mr, a, b, c, N):
    '''With the feature sizes in three dimensions a, b, and c, the relative mass mr and the array length N, return a numpy array of the size N that contains the corresponding photon wavelength of the excited states' energy.
    Input:
    mr (float): relative effective electron mass.
    a (float): Feature size in the first dimension (nm).
    b (float): Feature size in the second dimension (nm).
    c (float): Feature size in the Third dimension (nm).
    N (int): The length of returned array.
    Output:
    A (size N numpy array): The collection of the energy level wavelength.
    '''
    # Physical constants
    h = 6.626e-34  # Planck constant (J·s)
    c = 3e8         # Speed of light (m/s)
    
    # Get ground state wavelengths for each dimension
    lambda_a = ground_state_wavelength(a, mr)
    lambda_b = ground_state_wavelength(b, mr)
    lambda_c = ground_state_wavelength(c, mr)
    
    # Calculate ground state energies for each dimension (in joules)
    E_a = (h * c) / (lambda_a * 1e-9)
    E_b = (h * c) / (lambda_b * 1e-9)
    E_c = (h * c) / (lambda_c * 1e-9)
    
    # Generate first (N+1) smallest quadratic combinations of ground state energies
    sums_E = generate_quadratic_combinations(E_a, E_b, E_c, N + 1)
    
    # Compute incremental energies relative to ground state
    delta_E = sums_E - sums_E[0]
    
    # Remove zero element corresponding to ground state
    delta_E_nonzero = delta_E[1:]
    
    # Convert incremental energies to photon wavelengths (nanometers)
    lambda_photon_nm = (h * c / delta_E_nonzero) * 1e9
    
    return lambda_photon_nm
