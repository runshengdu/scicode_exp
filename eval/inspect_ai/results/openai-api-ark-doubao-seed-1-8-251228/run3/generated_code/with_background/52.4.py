import numpy as np
from scipy import integrate, optimize

def Schroed_deriv(y, r, l, En):
    '''Calculate the derivative of y given r, l and En
    Input 
    y=[u,u'], an list of float where u is the wave function at r, u' is the first derivative of u at r
    r: radius, float
    l: angular momentum quantum number, int
    En: energy, float
    Output
    Schroed: dy/dr=[u',u''] , an 1D array of float where u is the wave function at r, u' is the first derivative of u at r, u'' is the second derivative of u at r
    '''
    u, u_prime = y
    if r == 0:
        # Safeguard against division by zero (solver typically avoids r=0 in practice)
        u_double_prime = 0.0
    else:
        # Calculate second derivative using the dimensionless radial equation (Z=1)
        u_double_prime = (l * (l + 1) / (r ** 2) - 2.0 / r - En) * u
    return np.array([u_prime, u_double_prime])


def SolveSchroedinger(y0, En, l, R):
    '''Integrate the derivative of y within a certain radius
    Input 
    y0: Initial guess for function and derivative, list of floats: [u0, u0']
    En: energy, float
    l:  angular momentum quantum number, int
    R:  an 1d array (linespace) of radius (float)
    Output
    ur: the integration result, array of floats
    '''
    # Determine if the radius array is increasing or decreasing
    is_increasing = R[0] < R[-1]
    
    # Create integration array starting from large r to small r
    r_integrate = R[::-1] if is_increasing else R
    
    # Integrate the ODE using the provided derivative function
    y_result = integrate.odeint(Schroed_deriv, y0, r_integrate, args=(l, En))
    
    # Extract u values in the original radius array order
    if is_increasing:
        u_vals = y_result[::-1, 0]
    else:
        u_vals = y_result[:, 0]
    
    # Calculate the normalization integral using Simpson's rule
    norm_squared = integrate.simpson(u_vals ** 2, x=R)
    
    # Normalize the wavefunction
    ur = u_vals / np.sqrt(norm_squared)
    
    return ur


def Shoot(En, R, l, y0):
    '''Extrapolate u(0) based on results from SolveSchroedinger function
    Input 
    y0: Initial guess for function and derivative, list of floats: [u0, u0']
    En: energy, float
    R: an 1D array of (logspace) of radius; each element is a float
    l: angular momentum quantum number, int
    Output 
    f_at_0: Extrapolate u(0), float
    '''
    # Obtain the normalized radial wavefunction from the solver
    ur = SolveSchroedinger(y0, En, l, R)
    
    # Extract the first two points from the radial grid and corresponding wavefunction values
    r0, r1 = R[0], R[1]
    u0, u1 = ur[0], ur[1]
    
    # Scale the wavefunction values by dividing by r^l
    s0 = u0 / (r0 ** l)
    s1 = u1 / (r1 ** l)
    
    # Perform linear extrapolation to r=0
    if r1 == r0:
        # Avoid division by zero (logspace ensures distinct points, but handle edge case)
        f_at_0 = s0
    else:
        f_at_0 = (s0 * r1 - s1 * r0) / (r1 - r0)
    
    return f_at_0



def FindBoundStates(y0, R, l, nmax, Esearch):
    '''Input
    y0: Initial guess for function and derivative, list of floats: [u0, u0']
    R: an 1D array of (logspace) of radius; each element is a float
    l: angular momentum quantum number, int
    nmax: maximum number of bounds states wanted, int
    Esearch: energy mesh used for search, an 1D array of float
    Output
    Ebnd: a list, each element is a tuple containing the angular momentum quantum number (int) and energy (float) of all bound states found
    '''
    Ebnd = []
    if nmax <= 0 or len(Esearch) < 1:
        return Ebnd
    
    tolerance = 1e-6
    # Compute function values over the search energy mesh
    f_vals = np.array([Shoot(En, R, l, y0) for En in Esearch])
    
    # First, check for energies where f(En) is close to zero (within tolerance)
    zero_indices = np.where(np.abs(f_vals) < tolerance)[0]
    for idx in zero_indices:
        if len(Ebnd) >= nmax:
            break
        e_candidate = Esearch[idx]
        # Check for duplicates
        duplicate = any(np.isclose(e_candidate, e, atol=tolerance) for (ll, e) in Ebnd)
        if not duplicate:
            Ebnd.append((l, e_candidate))
    
    # Early return if we have enough bound states
    if len(Ebnd) >= nmax:
        return Ebnd
    
    # Check for sign changes between consecutive energy points to find intervals with roots
    mask = (f_vals[:-1] * f_vals[1:]) < 0
    sign_indices = np.where(mask)[0]
    
    # Extract existing energies for duplicate checking
    existing_energies = [e for (ll, e) in Ebnd]
    
    for idx in sign_indices:
        if len(Ebnd) >= nmax:
            break
        E_left = Esearch[idx]
        E_right = Esearch[idx + 1]
        f_left = f_vals[idx]
        f_right = f_vals[idx + 1]
        
        # Sanity check: ensure sign change exists
        if np.sign(f_left) == np.sign(f_right):
            continue
        
        # Define the function for root finding
        def root_func(en):
            return Shoot(en, R, l, y0)
        
        try:
            # Find the root using Brent's method
            e_bnd = optimize.brentq(root_func, E_left, E_right)
            # Check for duplicates with existing bound states
            duplicate = any(np.isclose(e_bnd, e, atol=tolerance) for e in existing_energies)
            if not duplicate:
                Ebnd.append((l, e_bnd))
                existing_energies.append(e_bnd)
        except (ValueError, optimize.nonlin.NoConvergence):
            # Skip if root finding fails (e.g., no root in interval, numerical issues)
            continue
    
    return Ebnd
