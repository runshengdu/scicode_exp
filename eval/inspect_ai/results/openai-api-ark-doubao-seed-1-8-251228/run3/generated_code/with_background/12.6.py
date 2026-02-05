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

    
    n = len(f_in)
    u = np.zeros_like(f_in)
    u[0] = u_at_0
    
    if n == 1:
        return u
    
    # Compute the second point using Taylor expansion up to u''(0)
    h = step
    h_sq = h ** 2
    u_second_deriv_0 = f_in[0] * u[0]
    u[1] = u[0] + h * up_at_0 + 0.5 * h_sq * u_second_deriv_0
    
    if n == 2:
        return u
    
    # Precompute constants for Numerov recursion
    h_sq_over_12 = h_sq / 12
    five_hso12 = 5 * h_sq_over_12
    
    # Iterate using the Numerov formula for third-order linear ODE
    for i in range(1, n - 1):
        term1 = 2 * u[i] * (1 + five_hso12 * f_in[i])
        term2 = u[i-1] * (1 - h_sq_over_12 * f_in[i-1])
        denominator = 1 - h_sq_over_12 * f_in[i+1]
        u[i+1] = (term1 - term2) / denominator
    
    return u


def compute_Schrod(energy, r_grid, l):
    '''Input 
    energy: a float
    r_grid: the radial grid; a 1D array of float
    l: angular momentum quantum number; an int
    Output
    ur_norm: normalized wavefunction u(x) at x = r
    '''
    # Calculate the f(r) array using the provided f_Schrod function
    f_in = f_Schrod(energy, l, r_grid)
    # Compute the step size as specified
    step = r_grid[0] - r_grid[1]
    # Solve the differential equation using the Numerov method
    u = Numerov(f_in, u_at_0=0.0, up_at_0=-1e-7, step=step)
    # Calculate the normalization integral using Simpson's rule
    integral = integrate.simpson(u ** 2, x=r_grid)
    # Compute the normalization factor (ensuring positive integral)
    norm_factor = np.sqrt(np.abs(integral))
    # Normalize the wavefunction
    ur_norm = u / norm_factor
    return ur_norm


def shoot(energy, r_grid, l):
    '''Input 
    energy: a float
    r_grid: the radial grid; a 1D array of float
    l: angular momentum quantum number; an int
    Output 
    f_at_0: float
    '''
    # Retrieve the normalized radial wavefunction u(r)
    ur_norm = compute_Schrod(energy, r_grid, l)
    
    # Get the last two grid points and corresponding u values (closest to r=0)
    r_near = r_grid[-2:]
    u_near = ur_norm[-2:]
    
    # Compute y = u / r^l for each near point
    y_near = u_near / (r_near ** l)
    
    # Unpack the points
    r1, r2 = r_near
    y1, y2 = y_near
    
    # Calculate linear extrapolation to r=0
    denominator = r2 - r1
    f_at_0 = (y1 * r2 - y2 * r1) / denominator
    
    return f_at_0


def find_bound_states(r_grid, l, energy_grid):
    '''Input
    r_grid: a 1D array of float
    l: angular momentum quantum number; int
    energy_grid: energy grid used for search; a 1D array of float
    Output
    bound_states: a list, each element is a tuple containing the angular momentum quantum number (int) and energy (float) of all bound states found
    '''
    bound_states = []
    # Filter to negative energies (bound states have negative energy)
    filtered_energies = [e for e in energy_grid if e < 0.0]
    if not filtered_energies:
        return bound_states
    # Sort energies in ascending order (most negative to least negative)
    sorted_energies = sorted(filtered_energies)
    
    # Define target function for root finding
    def target_func(E):
        return shoot(E, r_grid, l)
    
    # Iterate through consecutive energy pairs
    for i in range(len(sorted_energies) - 1):
        if len(bound_states) >= 10:
            break  # Reached maximum number of bound states
        e1 = sorted_energies[i]
        e2 = sorted_energies[i + 1]
        try:
            f1 = target_func(e1)
            f2 = target_func(e2)
        except Exception:
            # Skip if target_func raises an error (e.g., numerical issues)
            continue
        
        # Check if both function values are finite
        if not (np.isfinite(f1) and np.isfinite(f2)):
            continue
        
        # Check for sign change indicating root presence
        if f1 * f2 < 0:
            try:
                # Find root using Brentq method
                root_energy = optimize.brentq(target_func, e1, e2)
                # Add the bound state to the result list
                bound_states.append((l, root_energy))
            except (ValueError, RuntimeError):
                # Skip if root finding fails due to numerical issues
                continue
    return bound_states



def sort_states(bound_states):
    '''Input
    bound_states: a list of bound states found by the find_bound_states function, each element is a tuple containing the angular momentum quantum number (int) and energy (float)
    Output
    sorted_states: a list that contains the sorted bound_states tuples according to the following rules: State with lower energy will be in front. If two states have the same energy, the one with smaller angular momentum quantum number will be in front.
    '''
    # Sort by energy (ascending), then by angular momentum (ascending)
    sorted_states = sorted(bound_states, key=lambda x: (x[1], x[0]))
    return sorted_states
