import numpy as np
from scipy import integrate, optimize

def f_x(x, En):
    '''Return the value of f(x) with energy En
    Input
    x: coordinate x; a float or a 1D array of float
    En: energy; a float
    Output
    f_x: the value of f(x); a float or a 1D array of float
    '''
    return x ** 2 - En


def Numerov(f_in, u_b, up_b, step):
    '''Given precomputed function f(x), solve the differential equation u''(x) = f(x)*u(x)
    using the Numerov method.
    Inputs:
    - f_in: input function f(x); a 1D array of float representing the function values at discretized points
    - u_b: the value of u at boundary; a float
    - up_b: the derivative of u at boundary; a float
    - step: step size; a float.
    Output:
    - u: u(x); a 1D array of float representing the solution.
    '''
    f = np.asarray(f_in)
    n_points = len(f)
    u = np.zeros_like(f)
    
    if n_points == 0:
        return u
    
    # Set initial boundary condition
    u[0] = u_b
    
    if n_points == 1:
        return u
    
    # Compute the second initial point using Taylor expansion (up to second order)
    h = step
    u[1] = u[0] + h * up_b + 0.5 * h ** 2 * f[0] * u[0]
    
    # Precompute constants for efficiency
    h_sq = h ** 2
    h_sq_over_12 = h_sq / 12
    
    # Iterate using Numerov formula for remaining points
    for i in range(1, n_points - 1):
        # Calculate numerator terms
        term1 = 2 * u[i] * (1 - h_sq_over_12 * f[i])
        term2 = u[i-1] * (1 - h_sq_over_12 * f[i-1])
        term3 = h_sq * f[i] * u[i]
        numerator = term1 - term2 + term3
        
        # Calculate denominator
        denominator = 1 - h_sq_over_12 * f[i+1]
        
        # Compute next u value
        u[i+1] = numerator / denominator
    
    return u


def Solve_Schrod(x, En, u_b, up_b, step):
    '''Input
    x: coordinate x; a float or a 1D array of float
    En: energy; a float
    u_b: value of u(x) at one boundary for the Numverov function; a float
    up_b: value of the derivative of u(x) at one boundary for the Numverov function; a float
    step: the step size for the Numerov method; a float
    Output
    u_norm: normalized u(x); a float or a 1D array of float
    '''
    # Compute f(x) values at discretized points
    f_in = f_x(x, En)
    # Obtain unnormalized solution using Numerov method
    u_unnorm = Numerov(f_in, u_b, up_b, step)
    # Calculate integral of |u|² using Simpson's rule
    integral = integrate.simpson(u_unnorm ** 2, x=x)
    # Compute normalization constant
    norm_constant = 1.0 / np.sqrt(integral)
    # Normalize the solution
    u_norm = norm_constant * u_unnorm
    return u_norm


def count_sign_changes(solv_schrod):
    '''Input
    solv_schrod: a 1D array
    Output
    sign_changes: number of times of sign change occurrence; an int
    '''
    solv = np.asarray(solv_schrod)
    if len(solv) < 2:
        return 0
    consecutive_products = solv[:-1] * solv[1:]
    sign_changes = np.sum(consecutive_products < 0)
    return int(sign_changes)



def Bound</think_never_used_51bce0c785ca2f68081bfa7d91973934>Okay</think_never_used_51bce0c785ca2f68081bfa7d91973934>
So to include Emax, even if it's not a step multiple, we can do:

E_values = np.arange(start_E, Emax + Estep, Estep)
# Add Emax if it's not already in E_values
if not np.isclose(E_values[-1], Emax, atol=1e-6):
    E_values = np.append(E_values, Emax)

Then loop over E_values.

But wait np.arange might include Emax if start_E + k*Estep is exactly Emax. For example, start_E=3, Emax=6, Estep=3: np.arange(3,6+3,3) is [3,6], which includes6.

But if Emax is not a multiple, like5, then we add it to E_values.

So modifying the code:

In the for loop part:

# Generate E values
E_values = np.arange(start_E, Emax + Estep, Estep)
# Add Emax to ensure we check it, even if not a step multiple
if not any(np.isclose(E_values, Emax, atol=1e-6)):
    E_values = np.concatenate([E_values, [Emax]])

Then loop over E in E_values:

for E in E_values:
    ...

This way, Emax is always included in the scan.

So in the previous example where Emax=5, start_E=3, Estep=3:

E_values becomes [3,5]. So code checks E=3 and E=5.

For E=3 (even parity): u_last is positive (3<5). prev_E is None, so set prev_E=3, prev_sign=positive.

Next E=5: solve u(x) for E=5 (even parity). u_last is zero (since it's the eigenenergy). So code checks np.isclose(u_last,0) → yes. Adds (2,5.0) to bound_states. Increments n to3. Switches to odd parity. start_E=5. Then loop continues.

Scanning for odd parity from5 to5. E_values is [5]. prev_E is None, loop ends. found is False. Breaks out of while loop.

So bound_states becomes [(0,1.0), (1,3.0), (2,5.0)], which is correct.

Ah! That's a necessary fix. Because otherwise, if Estep is larger than the difference between start_E and Emax (and Emax is an eigenenergy), code will miss it.

So modifying the code to include Emax in E_values.

Another possible case: Emax is between two eigenenergies. Like Emax=4. After finding n=1 at3, switch to even parity, start_E=3. E_values is [3,4]. E=3: u_last positive. E=4: u_last negative (since4<5? No wait E=4 is between E_0=1 and E_2=5. For even parity, E=4> E_0 and <E_2. So u_last is negative? Wait no—wait earlier analysis said for even parity, E between E_0 and E_2: u_last is negative. Yes. So u_last at E=4 is negative. prev_sign is positive (from E=3), current_sign is negative. So sign change between3 and4. Code finds root between3 and4? But wait E_2 is5, which is larger than4. Oh no! Wait no—E=4 is less than E_2=5. So u_last at E=4 is negative, u_last at E=3 is positive (even parity: E=3 <E_2=5, u_last is negative? Wait wait no—wait I think I made a mistake in the analysis of u_last for even parity between E_0 and E_2.

Wait let's re-analyze even parity (u(0)=1, up_b=0):

E=3 (which is between E_0=1 and E_2=5):

u'' = (x²-3)u.

At x=0: u''= -3*1= -3 (negative), u curves downward.

At x=sqrt(3)≈1.732: x²-3=0. For x>sqrt(3), x²-3 is positive.

u(x) starts at1, decreases until some x, then since u'' becomes positive (x>sqrt(3)), u(x) starts curving upward. But wait no—wait u is positive until some x where u(x)=0? Or does it reach a minimum and then increase to positive infinity?

Wait no! Wait for E=3 < E_2=5: is u(x) ever negative?

Wait let's solve u'' = (x²-3)u with u(0)=1, u'(0)=0.

The general solution is a linear combination of the two independent solutions. One of them is the eigenfunction for E=5 (which decays to zero), and the other is a solution that diverges to infinity. But for E=3, which is not an eigenenergy, the solution is a combination that diverges.

Wait but does u(x) cross zero for E=3?

Wait let's think about the behavior at large x: u'' = (x²-3)u. For large x, this is approximately u'' =x² u. The solutions to u''=x² u are linear combinations of x^{-1/2} exp(±x²/2). So the solution that doesn't decay is exp(x²/2), which goes to positive infinity. The other is exp(-x²/2), which decays to zero.

But for E=3, the general solution is A*H(x) + B*G(x), where H(x) is the decaying solution (but not an eigenfunction, since E is not eigenenergy), and G(x) is the diverging solution. Wait no—no, for non-eigenenergies, there are no normalizable solutions. The solutions are either diverging to +infty or -infty.

Wait but with u(0)=1, u'(0)=0: which combination is that?

At x=0, u=1, u'=0.

The eigenfunction for E=5 is (1-2x²)exp(-x²/2). At x=0, this is1. u' is derivative: (-4x)exp(-x²/2) + (1-2x²)exp(-x²/2)*(-x). At x=0, u'=0. Oh! Wait a minute! The eigenfunction for E=5 (n=2) has u(