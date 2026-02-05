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
    # Constants from background
    r_B = 0.529  # Bohr radius in Å
    E0 = 13.6    # Rydberg energy in eV
    Z = 1        # Atomic number as specified
    
    # Calculate dimensionless variables
    x = r_grid / r_B
    eps = energy / E0
    
    # Compute f(x) in dimensionless form
    f_x = l * (l + 1) / (x ** 2) - 2 * Z / x - eps
    
    # Convert to f(r) for the original variable r
    f_r = f_x / (r_B ** 2)
    
    return f_r


def Numerov(f_in, u_at_0, up_at_0, step):
    '''Given precomputed function f(r), solve the differential equation u''(r) = f(r)*u(r)
    using the Numerov method.
    Inputs:
    - f_in: input function f(r); a 1D array of float representing the function values at discretized points.
    - u_at_0: the value of u at r = 0; a float.
    - up_at_0: the derivative of u at r = 0; a float.
    - step: step size; a float.
    Output:
    - u: the integration results at each point in the radial grid; a 1D array of float.
    '''
    f_in = np.asarray(f_in)
    n = len(f_in)
    u = np.zeros(n, dtype=np.float64)
    u[0] = u_at_0
    
    if n == 1:
        return u
    
    h = step
    # Compute u[1] using Taylor expansion up to O(h^3)
    u1 = u_at_0 + h * up_at_0 + 0.5 * h**2 * f_in[0] * u_at_0
    u[1] = u1
    
    if n == 2:
        return u
    
    # Iterate using direct Numerov recursion for remaining points
    for i in range(1, n - 1):
        numerator = 24 * u[i] - 12 * u[i-1] + (h**2) * (10 * f_in[i] * u[i] + f_in[i-1] * u[i-1])
        denominator = 12 - (h**2) * f_in[i+1]
        u[i+1] = numerator / denominator
    
    return u


def compute_Schrod(energy, r_grid, l):
    '''Input 
    energy: a float
    l: angular momentum quantum number; an int
    r_grid: the radial grid; a 1D array of float
    Output
    ur_norm: normalized wavefunction u(x) at x = r
    '''
    # Reverse the radial grid to integrate from the largest radius inward
    r_reversed = r_grid[::-1]
    # Compute the f(r) values for the reversed grid
    f_reversed = f_Schrod(energy, l, r_reversed)
    # Calculate the step size as specified (r_grid[0] - r_grid[1])
    step = r_grid[0] - r_grid[1]
    # Solve the differential equation using the Numerov method
    u_reversed = Numerov(f_reversed, u_at_0=0.0, up_at_0=-1e-7, step=step)
    # Reverse the solution to match the original grid order
    u_original = u_reversed[::-1]
    # Compute the normalization integral using Simpson's rule
    integral = integrate.simpson(u_original ** 2, x=r_grid)
    # Normalize the wavefunction
    ur_norm = u_original / np.sqrt(integral)
    
    return ur_norm


def shoot(energy, r_grid, l):
    '''Input 
    energy: a float
    r_grid: the radial grid; a 1D array of float
    l: angular momentum quantum number; an int
    Output 
    f_at_0: float
    '''
    # Get normalized radial wavefunction u(r)
    ur_norm = compute_Schrod(energy, r_grid, l)
    
    # Extract first two grid points and corresponding u values
    r0, r1 = r_grid[0], r_grid[1]
    u0, u1 = ur_norm[0], ur_norm[1]
    
    # Divide by r^l for each point
    v0 = u0 / (r0 ** l)
    v1 = u1 / (r1 ** l)
    
    # Linear extrapolation to r=0 (y-intercept of the line through (r0, v0) and (r1, v1))
    f_at_0 = (v0 * r1 - v1 * r0) / (r1 - r0)
    
    return f_at_0


def find_bound_states(r_grid, l, energy_grid):
    '''Input
    r_grid: a 1D array of float
    l: angular momentum quantum number; int
    energy_grid: energy grid used for search; a 1D array of float
    Output
    bound_states: a list, each element is a tuple containing the angular momentum quantum number (int) and energy (float) of all bound states found
    '''
    # Compute shoot function values for all energies in the grid
    shoot_vals = np.array([shoot(energy, r_grid, l) for energy in energy_grid])
    
    bound_states = []
    near_zero_threshold = 1e-10
    
    # First, check for exact near-zero values (bound states exactly on the grid)
    exact_zero_indices = np.where(np.abs(shoot_vals) < near_zero_threshold)[0]
    for idx in exact_zero_indices:
        if len(bound_states) >= 10:
            break
        bound_energy = energy_grid[idx]
        bound_states.append((l, bound_energy))
    
    # Find intervals where shoot function changes sign
    sign_changes = np.where(np.diff(np.sign(shoot_vals)) != 0)[0]
    
    # Process each sign change interval to find roots
    for idx in sign_changes:
        if len(bound_states) >= 10:
            break
        
        a = energy_grid[idx]
        b = energy_grid[idx + 1]
        f_a = shoot_vals[idx]
        f_b = shoot_vals[idx + 1]
        
        # Skip if signs are the same (sanity check) or if we already added exact zero
        if np.sign(f_a) == np.sign(f_b):
            continue
        if np.abs(f_a) < near_zero_threshold or np.abs(f_b) < near_zero_threshold:
            continue
        
        # Use Brentq to find the bound state energy in the interval
        try:
            bound_energy = optimize.brentq(lambda e: shoot(e, r_grid, l), a, b)
            bound_states.append((l, bound_energy))
        except ValueError:
            # Skip if root-finding fails (e.g., numerical issues)
            continue
    
    # Remove duplicate energies (if any) by rounding to 10 decimal places
    seen = set()
    unique_bound_states = []
    for state in bound_states:
        rounded_energy = round(state[1], 10)
        if rounded_energy not in seen:
            seen.add(rounded_energy)
            unique_bound_states.append(state)
            if len(unique_bound_states) >= 10:
                break
    
    return unique_bound_states



def sort_states(bound_states):
    '''Input
    bound_states: a list of bound states found by the find_bound_states function, each element is a tuple containing the angular momentum quantum number (int) and energy (float)
    Output
    sorted_states: a list that contains the sorted bound_states tuples according to the following rules: State with lower energy will be in front. If two states have the same energy, the one with smaller angular momentum quantum number will be in front.
    '''
    # Sort by energy (ascending) first, then by angular momentum (ascending)
    sorted_states = sorted(bound_states, key=lambda x: (x[1], x[0]))
    return sorted_states
