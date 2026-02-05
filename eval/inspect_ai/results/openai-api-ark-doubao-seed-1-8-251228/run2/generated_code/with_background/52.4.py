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
    # Calculate second derivative using the dimensionless Schrodinger equation (Z=1)
    u_double_prime = (l * (l + 1) / (r ** 2) - 2.0 / r - En) * u
    # Return the first-order system derivative as a 1D array
    return np.array([u_prime, u_double_prime])


def SolveSchroedinger(y0, En, l, R):
    '''Integrate the derivative of y within a certain radius
    Input 
    y0: Initial guess for function and derivative, list of floats: [u0, u0']
    En: energy, float
    l:  angular momentum quantum number, int
    R:  an 1d array (linespace) of radius (float)
    Output
    ur: the integration result, float
    '''
    # Determine integration order to start from large r
    if R[0] < R[-1]:
        # Reverse R to integrate from largest to smallest radius
        r_integrate = R[::-1]
        # Perform integration
        y_sol = integrate.odeint(Schroed_deriv, y0, r_integrate, args=(l, En))
        # Reverse solution to match original R order
        ur_before_norm = y_sol[:, 0][::-1]
    else:
        # R is already decreasing, integrate directly
        y_sol = integrate.odeint(Schroed_deriv, y0, R, args=(l, En))
        ur_before_norm = y_sol[:, 0]
    
    # Normalize using Simpson's rule
    integral = integrate.simpson(ur_before_norm ** 2, x=R)
    ur_normalized = ur_before_norm / np.sqrt(integral)
    
    return ur_normalized


def Shoot(En, R, l, y0):
    '''Extrapolate u(0) based on results from SolveSchroedinger function
    Input 
    y0: Initial guess for function and derivative, list of floats: [u0, u0']
    En: energy, float
    l:  angular momentum quantum number, int
    R:  an 1D array of (logspace) of radius; each element is a float
    Output 
    f_at_0: Extrapolate u(0), float
    '''
    # Obtain the normalized radial wavefunction
    ur = SolveSchroedinger(y0, En, l, R)
    
    # Extract the first two grid points (smallest radii)
    r1, r2 = R[0], R[1]
    u1, u2 = ur[0], ur[1]
    
    # Normalize wavefunction values by r^l
    f1 = u1 / (r1 ** l)
    f2 = u2 / (r2 ** l)
    
    # Linear extrapolation to r=0
    f_at_0 = (f1 * r2 - f2 * r1) / (r2 - r1)
    
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
    
    # Sort energy mesh and filter out non-negative energies (bound states have negative ε)
    sorted_E = np.sort(Esearch)
    mask = sorted_E < 0
    sorted_E = sorted_E[mask]
    if len(sorted_E) < 2:
        return Ebnd
    
    # Compute shooting function values for sorted energies
    f_vals = np.array([Shoot(En, R, l, y0) for En in sorted_E])
    
    # Iterate through consecutive energy pairs to find sign changes indicating roots
    for i in range(len(sorted_E) - 1):
        E1, E2 = sorted_E[i], sorted_E[i+1]
        f1, f2 = f_vals[i], f_vals[i+1]
        
        # Skip if both values are zero to avoid duplicate roots
        if np.isclose(f1, 0) and np.isclose(f2, 0):
            if not any(np.isclose(E1, e[1]) for e in Ebnd):
                Ebnd.append((l, E1))
                if len(Ebnd) >= nmax:
                    return Ebnd
            continue
        
        # Add exact root if f1 is zero and not already in Ebnd
        if np.isclose(f1, 0):
            if not any(np.isclose(E1, e[1]) for e in Ebnd):
                Ebnd.append((l, E1))
                if len(Ebnd) >= nmax:
                    return Ebnd
            continue
        
        # Check for sign change between E1 and E2
        if np.sign(f1) != np.sign(f2):
            # Define the function for root finding
            def shoot_energy_func(En):
                return Shoot(En, R, l, y0)
            
            # Attempt to find root using Brentq method
            try:
                root_result = optimize.root_scalar(
                    shoot_energy_func,
                    bracket=[E1, E2],
                    method='brentq'
                )
                # Add converged root if not already present
                if root_result.converged and not any(np.isclose(root_result.root, e[1]) for e in Ebnd):
                    Ebnd.append((l, root_result.root))
                    if len(Ebnd) >= nmax:
                        return Ebnd
            except ValueError:
                # Skip invalid bracket cases (numerical noise)
                continue
    
    return Ebnd
