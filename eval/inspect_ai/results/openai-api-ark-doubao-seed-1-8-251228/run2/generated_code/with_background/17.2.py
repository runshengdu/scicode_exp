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



def integrate_DOS(energy, energy_vertices):
    '''Input:
    energy: a float number representing the energy value at which the density of states will be integrated
    energy_vertices: a list of float numbers representing the energy values at the four vertices of a tetrahedron when implementing the linear tetrahedron method
    Output:
    result: a float number representing the integration results of the density of states
    '''
    # Sort vertex energies to ensure ε₁ < ε₂ < ε₃ < ε₄
    sorted_verts = sorted(energy_vertices)
    eps1, eps2, eps3, eps4 = sorted_verts
    E = energy

    # Case 1: E is below or equal to the lowest vertex energy - no contribution
    if E <= eps1:
        return 0.0
    # Case 5: E is above or equal to the highest vertex energy - no contribution
    elif E >= eps4:
        return 0.0
    # Case 2: E is between the lowest and second lowest vertex energies
    elif eps1 < E <= eps2:
        delta1 = E - eps1
        # Denominator D₁ = (ε₂-ε₁)(ε₃-ε₁)(ε₄-ε₁)
        D1 = (eps2 - eps1) * (eps3 - eps1) * (eps4 - eps1)
        result = 3 * (delta1 ** 2) / D1
        return result
    # Case 3: E is between the second and third lowest vertex energies
    elif eps2 < E <= eps3:
        delta1 = E - eps1
        delta2 = E - eps2
        # Denominators D₁ and D₂
        D1 = (eps2 - eps1) * (eps3 - eps1) * (eps4 - eps1)
        D2 = (eps1 - eps2) * (eps3 - eps2) * (eps4 - eps2)
        term1 = (delta1 ** 2) / D1
        term2 = (delta2 ** 2) / D2
        result = 3 * (term1 + term2)
        return result
    # Case 4: E is between the third and fourth lowest vertex energies
    elif eps3 < E < eps4:
        delta1 = E - eps1
        delta2 = E - eps2
        delta3 = E - eps3
        # Denominators D₁, D₂, D₃
        D1 = (eps2 - eps1) * (eps3 - eps1) * (eps4 - eps1)
        D2 = (eps1 - eps2) * (eps3 - eps2) * (eps4 - eps2)
        D3 = (eps1 - eps3) * (eps2 - eps3) * (eps4 - eps3)
        term1 = (delta1 ** 2) / D1
        term2 = (delta2 ** 2) / D2
        term3 = (delta3 ** 2) / D3
        result = 3 * (term1 + term2 + term3)
        return result
    # Fallback for unexpected cases (should not reach here with valid inputs)
    else:
        return 0.0
