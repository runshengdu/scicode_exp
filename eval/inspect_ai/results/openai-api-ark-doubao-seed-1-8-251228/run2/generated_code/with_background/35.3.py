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
    m0 = 9.109e-31
    c = 3e8
    h = 6.626e-34
    lmbd = (8 * c * mr * m0 * (L ** 2) * 1e-9) / h
    return lmbd




def generate_quadratic_combinations(x, y, z, N):
    '''With three numbers given, return an array with the size N that contains the smallest N numbers which are quadratic combinations of the input numbers.
    Input:
    x (float): The first number.
    y (float): The second number.
    z (float): The third number.
    N (int): The number of smallest combinations to return.
    Output:
    C (size N numpy array): The collection of the quadratic combinations.
    '''
    result = []
    visited = set()
    heap = []
    
    # Initialize with i=1, j=1, k=1
    initial_sum = x + y + z
    heapq.heappush(heap, (initial_sum, 1, 1, 1))
    visited.add((1, 1, 1))
    
    while len(result) < N:
        current_sum, i, j, k = heapq.heappop(heap)
        result.append(current_sum)
        
        # Generate next possible states
        # State 1: i+1, j, k
        if (i + 1, j, k) not in visited:
            new_sum = current_sum + (2 * i + 1) * x
            heapq.heappush(heap, (new_sum, i + 1, j, k))
            visited.add((i + 1, j, k))
        
        # State 2: i, j+1, k
        if (i, j + 1, k) not in visited:
            new_sum = current_sum + (2 * j + 1) * y
            heapq.heappush(heap, (new_sum, i, j + 1, k))
            visited.add((i, j + 1, k))
        
        # State 3: i, j, k+1
        if (i, j, k + 1) not in visited:
            new_sum = current_sum + (2 * k + 1) * z
            heapq.heappush(heap, (new_sum, i, j, k + 1))
            visited.add((i, j, k + 1))
    
    return np.array(result)



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
    # Step 1: Retrieve ground state wavelengths for each dimension
    lambda_a = ground_state_wavelength(a, mr)
    lambda_b = ground_state_wavelength(b, mr)
    lambda_c = ground_state_wavelength(c, mr)
    
    # Physical constants
    h = 6.626e-34
    c = 3e8
    
    # Step 2: Convert wavelengths to ground state energies (in Joules)
    E_a = (h * c) / (lambda_a * 1e-9)  # Convert nm to meters
    E_b = (h * c) / (lambda_b * 1e-9)
    E_c = (h * c) / (lambda_c * 1e-9)
    
    # Step 3: Get first N+1 smallest total energy combinations
    total_energies = generate_quadratic_combinations(E_a, E_b, E_c, N + 1)
    
    # Step 4: Calculate incremental energies relative to ground state
    delta_E = total_energies - total_energies[0]
    
    # Step 5: Remove zero energy (ground state) and convert to wavelength
    delta_E_nonzero = delta_E[1:]
    lambda_array = (h * c / delta_E_nonzero) * 1e9  # Convert Joules to nm
    
    return lambda_array
