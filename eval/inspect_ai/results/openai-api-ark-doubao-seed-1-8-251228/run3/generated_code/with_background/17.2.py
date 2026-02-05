import sympy as sp
import numpy as np

def init_eji_array(energy, energy_vertices):
    '''Initialize and populate a 5x5 array for storing e_ji variables, and map e_ji values
    to sympy symbols for later evaluation.
    Inputs:
    - energy: A float representing the energy level for density of states integration.
    - energy_vertices: A list of floats representing energy values at tetrahedron vertices.
    Outputs:
    - symbols: A dictionary mapping sympy symbol names to symbols.
    - value_map: A dictionary mapping symbols to their actual values (float).
    '''
    # Process energy values: ε0 = E (input energy), ε1-ε4 sorted vertex energies
    eps0 = energy
    sorted_vertices = sorted(energy_vertices)
    eps = [eps0] + sorted_vertices
    eps_array = np.array(eps)
    
    # Create 5x5 array of energy differences e_ji = ε_j - ε_i
    eji_array = eps_array[:, np.newaxis] - eps_array[np.newaxis, :]
    
    # Initialize symbol dictionary and value mapping
    symbols = {}
    value_map = {}
    
    # Create sympy symbols and map to their numerical values
    for j in range(5):
        for i in range(5):
            sym_name = f'e_{j}{i}'
            sym = sp.Symbol(sym_name)
            symbols[sym_name] = sym
            value_map[sym] = float(eji_array[j, i])
    
    return symbols, value_map



print('Hello, World!')
