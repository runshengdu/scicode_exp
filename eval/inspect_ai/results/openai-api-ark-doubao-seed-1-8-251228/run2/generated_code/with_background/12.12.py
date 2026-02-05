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


def calculate_charge_density(bound_states, r_grid, Z):
    '''Input
    bound_states: bound states found using the find_bound_states function; a list of tuples
    r_grid: the radial grid; a 1D array of float
    Z: atomic number; int
    Output
    charge_density: the calculated charge density coming from the bound states; 1D array of float
    '''
    # Sort bound states by energy (ascending), then angular momentum (ascending)
    sorted_states = sort_states(bound_states)
    
    # Initialize total charge density to zero
    charge_density = np.zeros_like(r_grid, dtype=np.float64)
    
    # Track number of electrons assigned to orbitals
    assigned_electrons = 0
    
    for state in sorted_states:
        if assigned_electrons >= Z:
            break
        
        l, energy = state
        # Calculate orbital degeneracy: 2*(2l+1) (spin + magnetic quantum numbers)
        degeneracy = 2 * (2 * l + 1)
        remaining = Z - assigned_electrons
        
        # Determine Fermi factor and number of electrons to add to this orbital
        if remaining >= degeneracy:
            fermi_factor = 1.0
            electrons_added = degeneracy
        else:
            fermi_factor = remaining / degeneracy
            electrons_added = remaining
        
        # Get normalized radial wavefunction u(r)
        ur_norm = compute_Schrod(energy, r_grid, l)
        
        # Compute (u(r)/r)^2, handling potential division by zero at r=0
        ur_over_r = ur_norm / r_grid
        nan_mask = np.isnan(ur_over_r)
        if np.any(nan_mask):
            # Interpolate to fill NaN values using valid nearby points
            valid_r = r_grid[~nan_mask]
            valid_ur_over_r = ur_over_r[~nan_mask]
            ur_over_r[nan_mask] = np.interp(r_grid[nan_mask], valid_r, valid_ur_over_r)
        
        # Charge density for fully filled orbital (using e=1 in atomic charge units)
        full_orbital_density = (-1) * (degeneracy / (4 * np.pi)) * (ur_over_r ** 2)
        
        # Calculate contribution from partially or fully filled orbital
        contribution = fermi_factor * full_orbital_density
        
        # Add to total charge density
        charge_density += contribution
        
        # Update count of assigned electrons
        assigned_electrons += electrons_added
    
    return charge_density


def calculate_HartreeU(charge_density, u_at_0, up_at_0, step, r_grid, Z):
    '''Input
    charge_density: the calculated charge density of the bound states; 1D array of float
    u_at_0: the value of u at r = 0; float
    up_at_0: the derivative of u at r = 0; float
    step: step size; float.
    r_grid: the radial grid; a 1D array of float
    Z: atomic number; int
    Output
    x: the HartreeU term with U(r)=V_H(r)r; 1D array of float
    '''
    # Compute the right-hand side of the differential equation U''(r) = -8π r ρ(r)
    u_rhs = -8 * np.pi * r_grid * charge_density
    
    n = len(r_grid)
    if n == 0:
        return np.array([])
    
    # Initialize arrays for particular solution U_p and w
    U_p = np.zeros(n, dtype=np.float64)
    w = np.zeros(n, dtype=np.float64)
    
    # Set initial conditions for U_p
    U_p[0] = u_at_0
    
    # Compute w[0]
    w[0] = U_p[0] - (step ** 2 / 12) * u_rhs[0]
    
    if n == 1:
        # Handle single-point grid case
        if r_grid[0] == 0:
            return np.array([0.0])
        alpha = (2 * Z - U_p[0]) / r_grid[0]
        return U_p + alpha * r_grid
    
    # Compute U_p[1] using Taylor expansion up to O(h^3)
    x_double_prime_0 = u_rhs[0]
    U_p[1] = u_at_0 + step * up_at_0 + 0.5 * step ** 2 * x_double_prime_0
    
    # Compute w[1]
    w[1] = U_p[1] - (step ** 2 / 12) * u_rhs[1]
    
    # Iterate using Numerov recursion to compute w for i >=2
    for i in range(1, n - 1):
        w[i + 1] = 2 * w[i] - w[i - 1] + step ** 2 * u_rhs[i]
    
    # Recompute U_p from w using the Numerov transformation (vectorized operation)
    U_p = w + (step ** 2 / 12) * u_rhs
    
    # Calculate alpha to satisfy the boundary condition U(∞) = 2Z
    r_max = r_grid[-1]
    U_p_max = U_p[-1]
    alpha = (2 * Z - U_p_max) / r_max
    
    # Construct the final solution U(r) = U_p(r) + alpha * r
    U = U_p + alpha * r_grid
    
    return U


def f_Schrod_Hartree(energy, r_grid, l, Z, hartreeU):
    '''Input 
    energy: float
    r_grid: the radial grid; a 1D array of float
    l: angular momentum quantum number; int
    Z: atomic number; int
    hartreeU: the values of the Hartree term U(r) in the form of U(r)=V_H(r)r, where V_H(r) is the actual Hartree potential term in the Schrodinger equation; a 1d array of float
    Output
    f_r: a 1D array of float 
    '''
    # Constants from background
    r_B = 0.529  # Bohr radius in Å
    E0 = 13.6    # Rydberg energy in eV
    
    # Calculate dimensionless variables
    x = r_grid / r_B
    eps = energy / E0
    
    # Compute f(x) in dimensionless form using the second equation from background
    f_x = l * (l + 1) / (x ** 2) + (hartreeU - 2 * Z) / x - eps
    
    # Convert to f(r) for the original variable r
    f_r = f_x / (r_B ** 2)
    
    return f_r


def compute_Schrod_Hartree(energy, r_grid, l, Z, hartreeU):
    '''Input 
    energy: float
    r_grid: the radial grid; a 1D array of float
    l: angular momentum quantum number; int
    Z: atomic number; int
    hartreeU: the values of the Hartree term U(r) in the form of U(r)=V_H(r)r, where V_H(r) is the actual Hartree potential term in the Schrodinger equation; a 1d array of float
    Output
    ur_norm: normalized wavefunction u(x) at x = r
    '''
    # Reverse the radial grid to integrate from the largest radius inward
    r_reversed = r_grid[::-1]
    # Reverse HartreeU to match the reversed radial grid
    hartreeU_reversed = hartreeU[::-1]
    
    # Compute f(r) for the reversed grid using f_Schrod_Hartree
    f_reversed = f_Schrod_Hartree(energy, r_reversed, l, Z, hartreeU_reversed)
    
    # Calculate step size as specified
    step = r_grid[0] - r_grid[1]
    
    # Solve the differential equation using the Numerov method
    u_reversed = Numerov(f_reversed, u_at_0=0.0, up_at_0=-1e-7, step=step)
    
    # Reverse the solution to match the original grid order
    u_original = u_reversed[::-1]
    
    # Compute normalization integral using Simpson's rule
    integral = integrate.simpson(u_original ** 2, x=r_grid)
    
    # Normalize the wavefunction
    ur_norm = u_original / np.sqrt(integral)
    
    return ur_norm


def extrapolate_polyfit(energy, r_grid, l, Z, hartreeU):
    '''Input 
    energy: float
    r_grid: the radial grid; a 1D array of float
    l: angular momentum quantum number; int
    Z: atomic number; int
    hartreeU: the values of the Hartree term U(r) in the form of U(r)=V_H(r)r, where V_H(r) is the actual Hartree potential term in the Schrodinger equation; a 1d array of float
    Output
    u0: the extrapolated value of u(r) at r=0; float
    '''
    # Get normalized radial wavefunction u(r)
    ur_norm = compute_Schrod_Hartree(energy, r_grid, l, Z, hartreeU)
    
    # Extract first four grid points and corresponding u values
    r_four = r_grid[:4]
    u_four = ur_norm[:4]
    
    # Compute v = u / r^l for each of the four points
    v_four = u_four / (r_four ** l)
    
    # Fit a 3rd-order polynomial to (r_four, v_four)
    coeffs = np.polyfit(r_four, v_four, 3)
    poly = np.poly1d(coeffs)
    
    # Evaluate the polynomial at r=0 to get the extrapolated v(0)
    v0 = poly(0.0)
    
    # Calculate u(0): for l>0, u(r) ~ r^{l+1} so u(0)=0; for l=0, u(0)=v0
    u0 = v0 if l == 0 else 0.0
    
    return u0





def find_bound_states_Hartree(r_grid, l, energy_grid, Z, hartreeU):
    '''Input
    r_grid: a 1D array of float
    l: angular momentum quantum number; int
    energy_grid: energy grid used for search; a 1D array of float
    Z: atomic number; int
    hartreeU: the values of the Hartree term U(r) in the form of U(r)=V_H(r)r, where V_H(r) is the actual Hartree potential term in the Schrodinger equation; a 1d array of float
    Output
    bound_states: a list, each element is a tuple containing the angular momentum quantum number (int) and energy (float) of all bound states found
    '''
    # Compute extrapolate_polyfit values for all energies in the grid
    shoot_vals = np.array([extrapolate_polyfit(energy, r_grid, l, Z, hartreeU) for energy in energy_grid])
    
    bound_states = []
    near_zero_threshold = 1e-10
    max_states = 10
    
    # Check for exact near-zero values (bound states exactly on the grid)
    exact_zero_indices = np.where(np.abs(shoot_vals) < near_zero_threshold)[0]
    for idx in exact_zero_indices:
        if len(bound_states) >= max_states:
            break
        bound_energy = energy_grid[idx]
        bound_states.append((l, bound_energy))
    
    # Find intervals where shoot_vals changes sign (indicates root crossing)
    sign_changes = np.where(np.diff(np.sign(shoot_vals)) != 0)[0]
    
    # Process each sign change interval to find roots using Brentq
    for idx in sign_changes:
        if len(bound_states) >= max_states:
            break
        
        a = energy_grid[idx]
        b = energy_grid[idx + 1]
        f_a = shoot_vals[idx]
        f_b = shoot_vals[idx + 1]
        
        # Skip intervals with same sign (sanity check) or exact zeros already processed
        if np.sign(f_a) == np.sign(f_b):
            continue
        if np.abs(f_a) < near_zero_threshold or np.abs(f_b) < near_zero_threshold:
            continue
        
        # Attempt to find root using Brentq
        try:
            bound_energy = optimize.brentq(
                lambda e: extrapolate_polyfit(e, r_grid, l, Z, hartreeU),
                a, b
            )
            bound_states.append((l, bound_energy))
        except ValueError:
            # Skip if root-finding fails (e.g., numerical instability or no root)
            continue
    
    # Remove duplicate energies by rounding to 10 decimal places
    seen = set()
    unique_bound_states = []
    for state in bound_states:
        rounded_energy = round(state[1], 10)
        if rounded_energy not in seen:
            seen.add(rounded_energy)
            unique_bound_states.append(state)
            if len(unique_bound_states) >= max_states:
                break
    
    return unique_bound_states
