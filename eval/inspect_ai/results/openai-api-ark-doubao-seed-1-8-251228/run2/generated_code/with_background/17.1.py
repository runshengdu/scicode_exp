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
    # Sort vertices to ensure ε₁ < ε₂ < ε₃ < ε₄ as required
    sorted_vertices = sorted(energy_vertices)
    # Create list of all energy values: ε₀=E, ε₁-ε₄ sorted vertices
    e = [energy] + sorted_vertices
    
    # Initialize and populate 5x5 e_ji array (e_ji = ε_j - ε_i)
    eji_array = np.zeros((5, 5), dtype=float)
    for j in range(5):
        for i in range(5):
            eji_array[j, i] = e[j] - e[i]
    
    # Create sympy symbols and value mappings
    symbols = {}
    value_map = {}
    for j in range(5):
        for i in range(5):
            sym_name = f'e_{j}_{i}'
            # Create real-valued sympy symbol (energy differences are real)
            sym = sp.Symbol(sym_name, real=True)
            symbols[sym_name] = sym
            value_map[sym] = eji_array[j, i]
    
    return symbols, value_map
