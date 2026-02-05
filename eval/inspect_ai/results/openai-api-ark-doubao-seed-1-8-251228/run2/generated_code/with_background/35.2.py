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
