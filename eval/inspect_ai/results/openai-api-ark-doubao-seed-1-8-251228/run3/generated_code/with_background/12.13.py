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


def calculate_charge_density(bound_states, r_grid, Z):
    '''Input
    bound_states: bound states found using the find_bound_states function; a list of tuples
    r_grid: the radial grid; a 1D array of float
    Z: atomic number; int
    Output
    charge_density: the calculated charge density coming from the bound states; 1D array of float
    '''
    # Sort the bound states according to the specified rules
    sorted_states = sort_states(bound_states)
    
    # Initialize charge density array with zeros
    charge_density = np.zeros_like(r_grid)
    
    num_electrons_placed = 0
    
    for l, energy in sorted_states:
        if num_electrons_placed >= Z:
            break
        
        # Calculate degeneracy of the current orbital
        degeneracy = 2 * (2 * l + 1)
        remaining_electrons = Z - num_electrons_placed
        
        # Determine the fermi factor
        if remaining_electrons >= degeneracy:
            fermi_factor = 1.0
            num_electrons_placed += degeneracy
        else:
            fermi_factor = remaining_electrons / degeneracy
            num_electrons_placed = Z
        
        # Get the normalized radial wavefunction u(r)
        u = compute_Schrod(energy, l, r_grid)
        
        # Calculate charge density contribution for a fully filled orbital
        full_contribution = - (degeneracy * u ** 2) / (4 * np.pi * r_grid ** 2)
        
        # Compute actual contribution using fermi factor
        contribution = fermi_factor * full_contribution
        
        # Add to total charge density
        charge_density += contribution
    
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
    # Compute the right-hand side of U''(r) = -8π r ρ(r)
    rhs = -8 * np.pi * r_grid * charge_density
    
    n = len(r_grid)
    if n == 1:
        # Handle single grid point case based on boundary conditions
        if np.isclose(r_grid[0], 0.0):
            return np.array([u_at_0])
        else:
            return np.array([2 * Z])
    
    # Reverse grid to ascending order (from r=0 to max r)
    r_asc = r_grid[::-1]
    rhs_asc = rhs[::-1]
    h = step
    
    # Initialize particular solution in ascending grid
    U_p_asc = np.zeros_like(r_asc)
    U_p_asc[0] = u_at_0
    # Calculate second point using Taylor expansion up to second derivative
    u_double_prime_0 = rhs_asc[0]
    U_p_asc[1] = U_p_asc[0] + h * up_at_0 + 0.5 * h**2 * u_double_prime_0
    
    # Precompute constants for Numerov recursion
    h_sq = h ** 2
    h_sq_over_12 = h_sq / 12
    
    # Initialize w array for Numerov transformation
    w_asc = np.zeros_like(r_asc)
    w_asc[0] = U_p_asc[0] - h_sq_over_12 * rhs_asc[0]
    w_asc[1] = U_p_asc[1] - h_sq_over_12 * rhs_asc[1]
    
    # Perform Numerov recursion
    for i in range(1, n - 1):
        # Update w array using Numerov recursion formula
        w_asc[i + 1] = 2 * w_asc[i] - w_asc[i - 1] + h_sq * rhs_asc[i]
        # Recover U_p from w array
        U_p_asc[i + 1] = w_asc[i + 1] + h_sq_over_12 * rhs_asc[i + 1]
    
    # Reverse back to original descending grid to get particular solution
    U_p = U_p_asc[::-1]
    
    # Calculate alpha to satisfy the boundary condition at infinity U(∞) = 2Z
    r_max = r_grid[0]
    alpha = (2 * Z - U_p[0]) / r_max
    
    # Construct the full solution including homogeneous term
    x = U_p + alpha * r_grid
    
    return x


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
    r_B = 0.529  # Bohr radius in Angstroms
    E0 = 13.6    # Rydberg energy in eV
    
    # Calculate each term of the f(r) expression
    term1 = l * (l + 1) / (r_grid ** 2)
    term2 = -2 * Z / (r_B * r_grid)
    
    # Precompute the constant (2m / ℏ²) which equals 1/(E0 * r_B²)
    two_m_over_hbar_sq = 1 / (E0 * r_B ** 2)
    term3 = hartreeU * two_m_over_hbar_sq / r_grid
    term4 = -energy * two_m_over_hbar_sq
    
    # Sum all terms to get the final f(r)
    f_r = term1 + term2 + term3 + term4
    
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
    # Calculate the f(r) array using the f_Schrod_Hartree function
    f_in = f_Schrod_Hartree(energy, r_grid, l, Z, hartreeU)
    
    # Compute the step size from the radial grid
    step = r_grid[0] - r_grid[1]
    
    # Solve the differential equation using the Numerov method
    u = Numerov(f_in, u_at_0=0.0, up_at_0=-1e-7, step=step)
    
    # Calculate the normalization integral using Simpson's rule
    integral = integrate.simpson(u ** 2, x=r_grid)
    
    # Compute normalization factor (ensuring positive value)
    norm_factor = np.sqrt(np.abs(integral))
    
    # Normalize the wavefunction
    ur_norm = u / norm_factor
    
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
    # Retrieve the normalized radial wavefunction u(r)
    ur_norm = compute_Schrod_Hartree(energy, r_grid, l, Z, hartreeU)
    
    # Get the last four grid points (closest to r=0) and corresponding u values
    r_near = r_grid[-4:]
    u_near = ur_norm[-4:]
    
    # Compute y = u / r^l for each near point
    y_near = u_near / (r_near ** l)
    
    # Fit a 3rd-order polynomial to (r_near, y_near)
    coeffs = np.polyfit(r_near, y_near, 3)
    
    # Create a polynomial function from the coefficients
    poly = np.poly1d(coeffs)
    
    # Extrapolate y to r=0, which corresponds to u(r)/r^l at r=0
    y0 = poly(0.0)
    
    # Calculate u0: for l=0, u0 = y0; for l>0, u0 = y0 * 0^l = 0
    if l == 0:
        u0 = y0
    else:
        u0 = 0.0
    
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
    bound_states = []
    
    # Filter to negative energies (bound states have negative energy)
    filtered_energies = [e for e in energy_grid if e < 0.0]
    if not filtered_energies:
        return bound_states
    
    # Sort energies in ascending order (most negative to least negative)
    sorted_energies = sorted(filtered_energies)
    
    # Define target function for root finding
    def target_func(E):
        return extrapolate_polyfit(E, r_grid, l, Z, hartreeU)
    
    # Iterate through consecutive energy pairs to search for roots
    for i in range(len(sorted_energies) - 1):
        if len(bound_states) >= 10:
            break  # Reached maximum number of bound states
        
        e1 = sorted_energies[i]
        e2 = sorted_energies[i + 1]
        
        try:
            f1 = target_func(e1)
            f2 = target_func(e2)
        except Exception:
            # Skip if target function encounters numerical issues
            continue
        
        # Skip if function values are not finite
        if not (np.isfinite(f1) and np.isfinite(f2)):
            continue
        
        # Check for sign change indicating a root between e1 and e2
        if f1 * f2 < 0:
            try:
                # Find root using Brent's method
                root_energy = optimize.brentq(target_func, e1, e2)
                bound_states.append((l, root_energy))
            except (ValueError, RuntimeError):
                # Skip if root finding fails due to numerical instability
                continue
    
    return bound_states



def calculate_charge_density_Hartree(bound_states, r_grid, Z, hartreeU):
    '''Input
    bound_states: bound states found using the find_bound_states function; a list of tuples
    r_grid: the radial grid; a 1D array of float
    Z: atomic number; int
    hartreeU: the values of the Hartree term U(r) in the form of U(r)=V_H(r)r, where V_H(r) is the actual Hartree potential term in the Schrodinger equation; a 1d array of float
    Output
    a tuple of the format (charge_density, total_energy), where:
        charge_density: the calculated charge density of the bound states; 1D array of float
        total_energy: the total energy of the bound states; float
    '''
    # Sort the bound states using the specified sorting function
    sorted_states = sort_states(bound_states)
    
    # Initialize charge density array and counters
    charge_density = np.zeros_like(r_grid)
    num_electrons_placed = 0
    total_energy = 0.0
    
    for l, energy in sorted_states:
        if num_electrons_placed >= Z:
            break
        
        # Calculate degeneracy of the current orbital
        degeneracy = 2 * (2 * l + 1)
        remaining_electrons = Z - num_electrons_placed
        
        # Determine the Fermi factor for partial filling
        if remaining_electrons >= degeneracy:
            fermi_factor = 1.0
            num_electrons_placed += degeneracy
        else:
            fermi_factor = remaining_electrons / degeneracy
            num_electrons_placed = Z
        
        # Get normalized radial wavefunction for the current state
        u = compute_Schrod_Hartree(energy, r_grid, l, Z, hartreeU)
        
        # Calculate charge density contribution
        r_squared = r_grid ** 2
        full_contribution = - (degeneracy * u ** 2) / (4 * np.pi * r_squared)
        charge_contribution = full_contribution * fermi_factor
        charge_density += charge_contribution
        
        # Accumulate total energy (in Rydberg units)
        total_energy += energy * degeneracy * fermi_factor
    
    return charge_density, total_energy
