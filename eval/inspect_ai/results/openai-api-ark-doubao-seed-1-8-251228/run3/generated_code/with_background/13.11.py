from numpy import zeros, linspace, exp, sqrt
import numpy as np

def partial_derivs_vec(fct, delta):
    '''Computes the partial derivatives of a scalar field in three dimensions using second-order finite differences.
    Parameters:
    -----------
    fct : numpy.ndarray
        A 3D array representing the scalar field values on the grid. Shape: (nx, ny, nz).
    delta : float
        The grid spacing or step size in each spatial direction.
    Returns:
    --------
    deriv_x : numpy.ndarray
        The partial derivative of the field with respect to the x direction (∂f/∂x).
    deriv_y : numpy.ndarray
        The partial derivative of the field with respect to the y direction (∂f/∂y).
    deriv_z : numpy.ndarray
        The partial derivative of the field with respect to the z direction (∂f/∂z).
    '''
    nx, ny, nz = fct.shape
    
    # Initialize derivative arrays with the same shape and dtype as input
    deriv_x = np.zeros_like(fct)
    deriv_y = np.zeros_like(fct)
    deriv_z = np.zeros_like(fct)
    
    # Compute derivative with respect to x (axis 0)
    # Interior points: second-order central difference
    deriv_x[1:-1, :, :] = (fct[2:, :, :] - fct[:-2, :, :]) / (2 * delta)
    # Boundary: x = 0 (forward one-sided second-order)
    deriv_x[0, :, :] = (-3 * fct[0, :, :] + 4 * fct[1, :, :] - fct[2, :, :]) / (2 * delta)
    # Boundary: x = nx-1 (backward one-sided second-order)
    deriv_x[-1, :, :] = (3 * fct[-1, :, :] - 4 * fct[-2, :, :] + fct[-3, :, :]) / (2 * delta)
    
    # Compute derivative with respect to y (axis 1)
    # Interior points: second-order central difference
    deriv_y[:, 1:-1, :] = (fct[:, 2:, :] - fct[:, :-2, :]) / (2 * delta)
    # Boundary: y = 0 (forward one-sided second-order)
    deriv_y[:, 0, :] = (-3 * fct[:, 0, :] + 4 * fct[:, 1, :] - fct[:, 2, :]) / (2 * delta)
    # Boundary: y = ny-1 (backward one-sided second-order)
    deriv_y[:, -1, :] = (3 * fct[:, -1, :] - 4 * fct[:, -2, :] + fct[:, -3, :]) / (2 * delta)
    
    # Compute derivative with respect to z (axis 2)
    # Interior points: second-order central difference
    deriv_z[:, :, 1:-1] = (fct[:, :, 2:] - fct[:, :, :-2]) / (2 * delta)
    # Boundary: z = 0 (forward one-sided second-order)
    deriv_z[:, :, 0] = (-3 * fct[:, :, 0] + 4 * fct[:, :, 1] - fct[:, :, 2]) / (2 * delta)
    # Boundary: z = nz-1 (backward one-sided second-order)
    deriv_z[:, :, -1] = (3 * fct[:, :, -1] - 4 * fct[:, :, -2] + fct[:, :, -3]) / (2 * delta)
    
    return deriv_x, deriv_y, deriv_z


def laplace(fct, delta):
    '''Computes the Laplacian of a scalar field in the interior of a 3D grid using second-order finite differences.
    This function calculates the Laplacian of a scalar field on a structured 3D grid using a central finite difference
    scheme. The output boundary values are set to zero, while the interior values are computed using the finite difference method.
    Parameters:
    -----------
    fct : numpy.ndarray
        A 3D array representing the scalar field values on the grid. Shape: (nx, ny, nz).
    delta : float
        The grid spacing or step size in each spatial direction.
    Returns:
    --------
    lap : numpy.ndarray
        A 3D array representing the Laplacian of the scalar field. Shape: (nx, ny, nz).
        The boundary values are set to zero, while the interior values are computed using the finite difference method.
    '''
    nx, ny, nz = fct.shape
    
    # Initialize Laplacian array with zeros
    lap = np.zeros_like(fct)
    
    # Compute Laplacian for interior points (not on any boundary)
    # Second-order central difference for each second partial derivative
    # x-component: ∂²f/∂x²
    d2x = (fct[2:, 1:-1, 1:-1] - 2 * fct[1:-1, 1:-1, 1:-1] + fct[:-2, 1:-1, 1:-1]) / (delta ** 2)
    # y-component: ∂²f/∂y²
    d2y = (fct[1:-1, 2:, 1:-1] - 2 * fct[1:-1, 1:-1, 1:-1] + fct[1:-1, :-2, 1:-1]) / (delta ** 2)
    # z-component: ∂²f/∂z²
    d2z = (fct[1:-1, 1:-1, 2:] - 2 * fct[1:-1, 1:-1, 1:-1] + fct[1:-1, 1:-1, :-2]) / (delta ** 2)
    
    # Sum the components to get the Laplacian at interior points
    lap[1:-1, 1:-1, 1:-1] = d2x + d2y + d2z
    
    return lap


def gradient(fct, delta):
    '''Computes the gradient of a scalar field in the interior of a 3D grid using second-order finite differences.
    Parameters:
    -----------
    fct : numpy.ndarray
        A 3D array representing the scalar field values on the grid. Shape: (nx, ny, nz).
    delta : float
        The grid spacing or step size in all spatial directions.
    Returns:
    --------
    grad_x : numpy.ndarray
        A 3D array representing the partial derivative of the field with respect to the x direction (∂f/∂x). 
        Shape: (nx, ny, nz). Boundary values are zeroed out.
    grad_y : numpy.ndarray
        A 3D array representing the partial derivative of the field with respect to the y direction (∂f/∂y). 
        Shape: (nx, ny, nz). Boundary values are zeroed out.
    grad_z : numpy.ndarray
        A 3D array representing the partial derivative of the field with respect to the z direction (∂f/∂z). 
        Shape: (nx, ny, nz). Boundary values are zeroed out.
    '''
    nx, ny, nz = fct.shape
    
    # Initialize derivative arrays with zeros
    grad_x = np.zeros_like(fct)
    grad_y = np.zeros_like(fct)
    grad_z = np.zeros_like(fct)
    
    # Compute ∂f/∂x for interior points along x-axis (axis 0) using central difference
    grad_x[1:-1, :, :] = (fct[2:, :, :] - fct[:-2, :, :]) / (2 * delta)
    
    # Compute ∂f/∂y for interior points along y-axis (axis 1) using central difference
    grad_y[:, 1:-1, :] = (fct[:, 2:, :] - fct[:, :-2, :]) / (2 * delta)
    
    # Compute ∂f/∂z for interior points along z-axis (axis 2) using central difference
    grad_z[:, :, 1:-1] = (fct[:, :, 2:] - fct[:, :, :-2]) / (2 * delta)
    
    return grad_x, grad_y, grad_z


def divergence(v_x, v_y, v_z, delta):
    '''Computes the divergence of a 3D vector field using second-order finite differences.
    Parameters:
    -----------
    v_x : numpy.ndarray
        A 3D array representing the x-component of the vector field. Shape: (nx, ny, nz).
    v_y : numpy.ndarray
        A 3D array representing the y-component of the vector field. Shape: (nx, ny, nz).
    v_z : numpy.ndarray
        A 3D array representing the z-component of the vector field. Shape: (nx, ny, nz).
    delta : float
        The grid spacing or step size in all spatial directions.
    Returns:
    --------
    div : numpy.ndarray
        A 3D array representing the divergence of the vector field. Shape: (nx, ny, nz).
        The boundary values are set to zero.
    '''
    nx, ny, nz = v_x.shape
    
    # Initialize divergence array with zeros
    div = np.zeros_like(v_x)
    
    # Compute the three components of divergence at interior points using central differences
    # ∂v_x/∂x
    term_x = (v_x[2:, 1:-1, 1:-1] - v_x[:-2, 1:-1, 1:-1]) / (2 * delta)
    # ∂v_y/∂y
    term_y = (v_y[1:-1, 2:, 1:-1] - v_y[1:-1, :-2, 1:-1]) / (2 * delta)
    # ∂v_z/∂z
    term_z = (v_z[1:-1, 1:-1, 2:] - v_z[1:-1, 1:-1, :-2]) / (2 * delta)
    
    # Sum the terms to get divergence at interior points
    div[1:-1, 1:-1, 1:-1] = term_x + term_y + term_z
    
    return div


def grad_div(A_x, A_y, A_z, delta):
    '''Computes the gradient of the divergence of a 3D vector field using second-order finite differences.
    Parameters:
    -----------
    A_x : numpy.ndarray
        A 3D array representing the x-component of the vector field. Shape: (nx, ny, nz).
    A_y : numpy.ndarray
        A 3D array representing the y-component of the vector field. Shape: (nx, ny, nz).
    A_z : numpy.ndarray
        A 3D array representing the z-component of the vector field. Shape: (nx, ny, nz).
    delta : float
        The grid spacing or step size in all spatial directions.
    Returns:
    --------
    grad_div_x : numpy.ndarray
        A 3D array representing the x-component of the gradient of divergence. Shape: (nx, ny, nz).
        The boundary values are set to zero.
    grad_div_y : numpy.ndarray
        A 3D array representing the y-component of the gradient of divergence. Shape: (nx, ny, nz).
        The boundary values are set to zero.
    grad_div_z : numpy.ndarray
        A 3D array representing the z-component of the gradient of divergence. Shape: (nx, ny, nz).
        The boundary values are set to zero.
    '''
    nx, ny, nz = A_x.shape
    
    # Initialize result arrays with zeros
    grad_div_x = np.zeros_like(A_x)
    grad_div_y = np.zeros_like(A_y)
    grad_div_z = np.zeros_like(A_z)
    
    # Compute x-component of gradient of divergence at full interior points
    term1_x = (A_x[2:, 1:-1, 1:-1] - 2 * A_x[1:-1, 1:-1, 1:-1] + A_x[:-2, 1:-1, 1:-1]) / (delta ** 2)
    term2_x = (A_y[2:, 2:, 1:-1] - A_y[:-2, 2:, 1:-1] - A_y[2:, :-2, 1:-1] + A_y[:-2, :-2, 1:-1]) / (4 * delta ** 2)
    term3_x = (A_z[2:, 1:-1, 2:] - A_z[:-2, 1:-1, 2:] - A_z[2:, 1:-1, :-2] + A_z[:-2, 1:-1, :-2]) / (4 * delta ** 2)
    grad_div_x[1:-1, 1:-1, 1:-1] = term1_x + term2_x + term3_x
    
    # Compute y-component of gradient of divergence at full interior points
    term1_y = (A_x[2:, 2:, 1:-1] - A_x[:-2, 2:, 1:-1] - A_x[2:, :-2, 1:-1] + A_x[:-2, :-2, 1:-1]) / (4 * delta ** 2)
    term2_y = (A_y[1:-1, 2:, 1:-1] - 2 * A_y[1:-1, 1:-1, 1:-1] + A_y[1:-1, :-2, 1:-1]) / (delta ** 2)
    term3_y = (A_z[1:-1, 2:, 2:] - A_z[1:-1, 2:, :-2] - A_z[1:-1, :-2, 2:] + A_z[1:-1, :-2, :-2]) / (4 * delta ** 2)
    grad_div_y[1:-1, 1:-1, 1:-1] = term1_y + term2_y + term3_y
    
    # Compute z-component of gradient of divergence at full interior points
    term1_z = (A_x[2:, 1:-1, 2:] - A_x[:-2, 1:-1, 2:] - A_x[2:, 1:-1, :-2] + A_x[:-2, 1:-1, :-2]) / (4 * delta ** 2)
    term2_z = (A_y[1:-1, 2:, 2:] - A_y[1:-1, 2:, :-2] - A_y[1:-1, :-2, 2:] + A_y[1:-1, :-2, :-2]) / (4 * delta ** 2)
    term3_z = (A_z[1:-1, 1:-1, 2:] - 2 * A_z[1:-1, 1:-1, 1:-1] + A_z[1:-1, 1:-1, :-2]) / (delta ** 2)
    grad_div_z[1:-1, 1:-1, 1:-1] = term1_z + term2_z + term3_z
    
    return grad_div_x, grad_div_y, grad_div_z

def __init__(self, n_grid, x_out):
    """Constructor sets up coordinates, memory for variables.
        The variables:
            mesh points:
                x: the x coordinate for each mesh grid
                y: the y coordinate for each mesh grid
                z: the z coordinate for each mesh grid
                t: the time coordinate of the simulation
                r: the distance to the origin for each mesh grid
            evolving fields:
                E_x: the x component of the field E
                E_y: the y componnet of the field E
                E_z: the z component of the field E
                A_x: the x component of the field A
                A_y: the y component of the field A
                A_z: the z component of the field A
                phi: the scalar potential field phi values
            monitor variables:
                constraint: the current constraint violation value from the evolving fields.
                
        """
    self.n_grid = n_grid
    self.n_vars = 7
    self.delta = float(x_out) / (n_grid - 2.0)
    delta = self.delta
    self.x = np.linspace(-self.delta * 0.5, x_out + 0.5 * self.delta, self.n_grid)[:, None, None]
    self.y = np.linspace(-self.delta * 0.5, x_out + 0.5 * self.delta, self.n_grid)[None, :, None]
    self.z = np.linspace(-self.delta * 0.5, x_out + 0.5 * self.delta, self.n_grid)[None, None, :]
    self.r = np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
    self.E_x = zeros((n_grid, n_grid, n_grid))
    self.E_y = zeros((n_grid, n_grid, n_grid))
    self.E_z = zeros((n_grid, n_grid, n_grid))
    self.A_x = zeros((n_grid, n_grid, n_grid))
    self.A_y = zeros((n_grid, n_grid, n_grid))
    self.A_z = zeros((n_grid, n_grid, n_grid))
    self.phi = zeros((n_grid, n_grid, n_grid))
    self.constraint = zeros((n_grid, n_grid, n_grid))
    self.t = 0.0

def symmetry(f_dot, x_sym, y_sym, z_sym):
    '''Computes time derivatives on inner boundaries from symmetry
    Parameters:
    -----------
    f_dot : numpy.ndarray
        A 3D array representing the time derivatives of the scalar field. Shape: (nx, ny, nz).
        This array will be updated in-place with symmetric boundary conditions applied.
    x_sym : float
        The symmetry factor to apply along the x-axis (typically -1 for antisymmetry, 1 for symmetry).
    y_sym : float
        The symmetry factor to apply along the y-axis (typically -1 for antisymmetry, 1 for symmetry).
    z_sym : float
        The symmetry factor to apply along the z-axis (typically -1 for antisymmetry, 1 for symmetry).
    Returns:
    --------
    f_dot : numpy.ndarray
        The same 3D array passed in as input, with updated values at the boundaries according to the symmetry conditions.
        Shape: (nx, ny, nz).
    '''
    # Apply symmetry to x-boundary (i=0, j>=1, k>=1)
    f_dot[0, 1:, 1:] = x_sym * f_dot[1, 1:, 1:]
    
    # Apply symmetry to y-boundary (i>=1, j=0, k>=1)
    f_dot[1:, 0, 1:] = y_sym * f_dot[1:, 1, 1:]
    
    # Apply symmetry to z-boundary (i>=1, j>=1, k=0)
    f_dot[1:, 1:, 0] = z_sym * f_dot[1:, 1:, 1]
    
    # Apply symmetry to x-y corner (i=0, j=0, k>=1)
    f_dot[0, 0, 1:] = x_sym * y_sym * f_dot[1, 1, 1:]
    
    # Apply symmetry to x-z corner (i=0, j>=1, k=0)
    f_dot[0, 1:, 0] = x_sym * z_sym * f_dot[1, 1:, 1]
    
    # Apply symmetry to y-z corner (i>=1, j=0, k=0)
    f_dot[1:, 0, 0] = y_sym * z_sym * f_dot[1:, 1, 1]
    
    # Apply symmetry to origin corner (i=0, j=0, k=0)
    f_dot[0, 0, 0] = x_sym * y_sym * z_sym * f_dot[1, 1, 1]
    
    return f_dot


def outgoing_wave(maxwell, f_dot, f):
    '''Computes time derivatives of fields from outgoing-wave boundary condition
    Parameters:
    -----------
    maxwell : object
        An object containing properties of the simulation grid, including:
        - `delta`: Grid spacing (step size) in all spatial directions.
        - `x`, `y`, `z`: 3D arrays representing the coordinate grids along the x, y, and z axes, respectively.
        - `r`: 3D array representing the grid radial distance from the origin.
    f_dot : numpy.ndarray
        A 3D array representing the time derivatives of the scalar field. Shape: (nx, ny, nz).
        This array will be updated in-place with the outgoing wave boundary condition applied.
    f : numpy.ndarray
        A 3D array representing the scalar field values on the grid. Shape: (nx, ny, nz).
    Returns:
    --------
    f_dot : numpy.ndarray
        The same 3D array passed in as input, with updated values at the outer boundaries according to the
        outgoing wave boundary condition. Shape: (nx, ny, nz).
    '''
    delta = maxwell.delta
    nx, ny, nz = f.shape
    
    x = maxwell.x
    y = maxwell.y
    z = maxwell.z
    r = maxwell.r

    # Handle outer x boundary (i = nx-1, all j, k)
    # Compute ∂x f (backward second-order)
    dx_x_outer = (3 * f[-1, :, :] - 4 * f[-2, :, :] + f[-3, :, :]) / (2 * delta)
    # Compute ∂y f
    dy_x_outer = np.zeros_like(f[-1, :, :])
    dy_x_outer[1:-1, :] = (f[-1, 2:, :] - f[-1, :-2, :]) / (2 * delta)
    dy_x_outer[0, :] = (-3 * f[-1, 0, :] + 4 * f[-1, 1, :] - f[-1, 2, :]) / (2 * delta)
    dy_x_outer[-1, :] = (3 * f[-1, -1, :] - 4 * f[-1, -2, :] + f[-1, -3, :]) / (2 * delta)
    # Compute ∂z f
    dz_x_outer = np.zeros_like(f[-1, :, :])
    dz_x_outer[:, 1:-1] = (f[-1, :, 2:] - f[-1, :, :-2]) / (2 * delta)
    dz_x_outer[:, 0] = (-3 * f[-1, :, 0] + 4 * f[-1, :, 1] - f[-1, :, 2]) / (2 * delta)
    dz_x_outer[:, -1] = (3 * f[-1, :, -1] - 4 * f[-1, :, -2] + f[-1, :, -3]) / (2 * delta)
    # Update f_dot for outer x boundary
    r_x = r[-1, :, :]
    term_x = (-f[-1, :, :] - (x[-1, :, :] * dx_x_outer + y[-1, :, :] * dy_x_outer + z[-1, :, :] * dz_x_outer)) / r_x
    f_dot[-1, :, :] = term_x

    # Handle outer y boundary (j = ny-1, all i, k)
    # Compute ∂x f
    dx_y_outer = np.zeros_like(f[:, -1, :])
    dx_y_outer[1:-1, :] = (f[2:, -1, :] - f[:-2, -1, :]) / (2 * delta)
    dx_y_outer[0, :] = (-3 * f[0, -1, :] + 4 * f[1, -1, :] - f[2, -1, :]) / (2 * delta)
    dx_y_outer[-1, :] = (3 * f[-1, -1, :] - 4 * f[-2, -1, :] + f[-3, -1, :]) / (2 * delta)
    # Compute ∂y f (backward second-order)
    dy_y_outer = (3 * f[:, -1, :] - 4 * f[:, -2, :] + f[:, -3, :]) / (2 * delta)
    # Compute ∂z f
    dz_y_outer = np.zeros_like(f[:, -1, :])
    dz_y_outer[:, 1:-1] = (f[:, -1, 2:] - f[:, -1, :-2]) / (2 * delta)
    dz_y_outer[:, 0] = (-3 * f[:, -1, 0] + 4 * f[:, -1, 1] - f[:, -1, 2]) / (2 * delta)
    dz_y_outer[:, -1] = (3 * f[:, -1, -1] - 4 * f[:, -1, -2] + f[:, -1, -3]) / (2 * delta)
    # Update f_dot for outer y boundary
    r_y = r[:, -1, :]
    term_y = (-f[:, -1, :] - (x[:, -1, :] * dx_y_outer + y[:, -1, :] * dy_y_outer + z[:, -1, :] * dz_y_outer)) / r_y
    f_dot[:, -1, :] = term_y

    # Handle outer z boundary (k = nz-1, all i, j)
    # Compute ∂x f
    dx_z_outer = np.zeros_like(f[:, :, -1])
    dx_z_outer[1:-1, :] = (f[2:, :, -1] - f[:-2, :, -1]) / (2 * delta)
    dx_z_outer[0, :] = (-3 * f[0, :, -1] + 4 * f[1, :, -1] - f[2, :, -1]) / (2 * delta)
    dx_z_outer[-1, :] = (3 * f[-1, :, -1] - 4 * f[-2, :, -1] + f[-3, :, -1]) / (2 * delta)
    # Compute ∂y f
    dy_z_outer = np.zeros_like(f[:, :, -1])
    dy_z_outer[:, 1:-1] = (f[:, 2:, -1] - f[:, :-2, -1]) / (2 * delta)
    dy_z_outer[:, 0] = (-3 * f[:, 0, -1] + 4 * f[:, 1, -1] - f[:, 2, -1]) / (2 * delta)
    dy_z_outer[:, -1] = (3 * f[:, -1, -1] - 4 * f[:, -2, -1] + f[:, -3, -1]) / (2 * delta)
    # Compute ∂z f (backward second-order)
    dz_z_outer = (3 * f[:, :, -1] - 4 * f[:, :, -2] + f[:, :, -3]) / (2 * delta)
    # Update f_dot for outer z boundary
    r_z = r[:, :, -1]
    term_z = (-f[:, :, -1] - (x[:, :, -1] * dx_z_outer + y[:, :, -1] * dy_z_outer + z[:, :, -1] * dz_z_outer)) / r_z
    f_dot[:, :, -1] = term_z

    return f_dot


def derivatives(maxwell, fields):
    '''Computes the time derivatives of electromagnetic fields according to Maxwell's equations in Lorentz Gauge.
    Parameters:
    -----------
    maxwell : object
        An object containing properties of the simulation grid and field values, including:
        - `A_x`, `A_y`, `A_z`: 3D arrays representing the vector potential components.
        - `E_x`, `E_y`, `E_z`: 3D arrays representing the electric field components.
        - `phi`: 3D array representing the scalar potential.
        - `delta`: Grid spacing (step size) in all spatial directions.
    fields : tuple of numpy.ndarray
        A tuple containing the current field values in the following order: `(E_x, E_y, E_z, A_x, A_y, A_z, phi)`.
        Each component is a 3D array of shape `(nx, ny, nz)`.
    Returns:
    --------
    tuple of numpy.ndarray
        A tuple containing the time derivatives of the fields in the following order:
        `(E_x_dot, E_y_dot, E_z_dot, A_x_dot, A_y_dot, A_z_dot, phi_dot)`. Each component is a 3D array of shape `(nx, ny, nz)`.
    '''
    # Unpack input fields
    E_x, E_y, E_z, A_x, A_y, A_z, phi = fields
    delta = maxwell.delta
    
    # ---------------- Compute time derivatives for E fields ----------------
    # Calculate gradient of divergence of vector potential A
    grad_div_x, grad_div_y, grad_div_z = grad_div(A_x, A_y, A_z, delta)
    
    # Calculate Laplacian of each vector potential component
    laplace_A_x = laplace(A_x, delta)
    laplace_A_y = laplace(A_y, delta)
    laplace_A_z = laplace(A_z, delta)
    
    # Compute interior values of E field time derivatives (assuming j=0)
    E_x_dot = grad_div_x - laplace_A_x
    E_y_dot = grad_div_y - laplace_A_y
    E_z_dot = grad_div_z - laplace_A_z
    
    # Apply inner boundary symmetry conditions
    # E_x: odd in x, even in y, even in z
    E_x_dot = symmetry(E_x_dot, x_sym=-1, y_sym=1, z_sym=1)
    # E_y: even in x, odd in y, even in z
    E_y_dot = symmetry(E_y_dot, x_sym=1, y_sym=-1, z_sym=1)
    # E_z: even in x, even in y, odd in z
    E_z_dot = symmetry(E_z_dot, x_sym=1, y_sym=1, z_sym=-1)
    
    # Apply outgoing wave boundary conditions to E fields
    E_x_dot = outgoing_wave(maxwell, E_x_dot, E_x)
    E_y_dot = outgoing_wave(maxwell, E_y_dot, E_y)
    E_z_dot = outgoing_wave(maxwell, E_z_dot, E_z)
    
    # ---------------- Compute time derivatives for A fields ----------------
    # Calculate gradient of scalar potential phi
    grad_phi_x, grad_phi_y, grad_phi_z = gradient(phi, delta)
    
    # Compute interior values of A field time derivatives
    A_x_dot = -E_x - grad_phi_x
    A_y_dot = -E_y - grad_phi_y
    A_z_dot = -E_z - grad_phi_z
    
    # Apply inner boundary symmetry conditions
    # A_x: odd in x, even in y, even in z (same as E_x)
    A_x_dot = symmetry(A_x_dot, x_sym=-1, y_sym=1, z_sym=1)
    # A_y: even in x, odd in y, even in z (same as E_y)
    A_y_dot = symmetry(A_y_dot, x_sym=1, y_sym=-1, z_sym=1)
    # A_z: even in x, even in y, odd in z (same as E_z)
    A_z_dot = symmetry(A_z_dot, x_sym=1, y_sym=1, z_sym=-1)
    
    # Apply outgoing wave boundary conditions to A fields
    A_x_dot = outgoing_wave(maxwell, A_x_dot, A_x)
    A_y_dot = outgoing_wave(maxwell, A_y_dot, A_y)
    A_z_dot = outgoing_wave(maxwell, A_z_dot, A_z)
    
    # ---------------- Compute time derivative for phi ----------------
    # Calculate divergence of vector potential A
    div_A = divergence(A_x, A_y, A_z, delta)
    
    # Compute interior values of phi time derivative
    phi_dot = -div_A
    
    # Apply inner boundary symmetry conditions (phi is scalar, even in all directions)
    phi_dot = symmetry(phi_dot, x_sym=1, y_sym=1, z_sym=1)
    
    # Apply outgoing wave boundary condition to phi
    phi_dot = outgoing_wave(maxwell, phi_dot, phi)
    
    return (E_x_dot, E_y_dot, E_z_dot, A_x_dot, A_y_dot, A_z_dot, phi_dot)


def update_fields(maxwell, fields, fields_dot, factor, dt):
    '''Updates all fields by adding a scaled increment of the time derivatives.
    Parameters:
    -----------
    maxwell : object
        An object representing the Maxwell simulation environment, including:
        - `n_vars`: Number of variables in the simulation (e.g., number of field components).
    fields : list of numpy.ndarray
        A list containing the current field values to be updated, in the following order:
        `[E_x, E_y, E_z, A_x, A_y, A_z, phi]`.
        Each field is a 3D array of shape `(nx, ny, nz)`.
    fields_dot : list of numpy.ndarray
        A list containing the time derivatives of the corresponding fields, in the same order as `fields`.
        Each derivative is a 3D array of shape `(nx, ny, nz)`.
    factor : float
        A scaling factor to be applied to the field updates. (useful when applying in the higher order time integrator like Runge-Kutta)
    dt : float
        Time step size.
    Returns:
    --------
    list of numpy.ndarray
        A list containing the updated fields in the same order as `fields`.
        Each updated field is a 3D array of shape `(nx, ny, nz)`.
    '''
    # Compute updated fields using scaled time derivatives
    new_fields = [field + factor * dt * field_dot for field, field_dot in zip(fields, fields_dot)]
    
    return new_fields



def crank_nicholson_step(maxwell, fields, dt):
    '''Performs a single iterative Crank-Nicholson (ICN) time step of size dt.
    This function implements the second-order ICN scheme equivalent to:
    k1 = derivatives(t0, f0)
    k2 = derivatives(t0+dt, f0 + k1*dt)
    k3 = derivatives(t0+dt, f0 + k2*dt)
    f(t0+dt) = f0 + (k1 + k3)*dt/2
    Parameters:
    -----------
    maxwell : object
        An object containing properties of the simulation grid and field values.
    fields : list of numpy.ndarray
        A list containing the current field values, in the following order:
        `[E_x, E_y, E_z, A_x, A_y, A_z, phi]`.
        Each field is a 3D array of shape `(nx, ny, nz)`.
    dt : float
        Time step size for the ICN step.
    Returns:
    --------
    list of numpy.ndarray
        Updated fields after the ICN step, in the same order as input.
    '''
    # Compute k1: time derivatives at current fields
    k1 = derivatives(maxwell, fields)
    # Compute f1 = f0 + k1*dt
    f1 = update_fields(maxwell, fields, k1, 1.0, dt)
    # Compute k2: time derivatives at f1
    k2 = derivatives(maxwell, f1)
    # Compute f2 = f0 + k2*dt
    f2 = update_fields(maxwell, fields, k2, 1.0, dt)
    # Compute k3: time derivatives at f2
    k3 = derivatives(maxwell, f2)
    # Compute component-wise sum of k1 and k3
    k1_plus_k3 = [k1_comp + k3_comp for k1_comp, k3_comp in zip(k1, k3)]
    # Update fields using the average of k1 and k3 scaled by dt/2
    new_fields = update_fields(maxwell, fields, k1_plus_k3, 0.5, dt)
    return new_fields


def stepper(maxwell, fields, courant, t_const):
    '''Executes an iterative Crank-Nicholson (ICN) step using a Runge-Kutta scheme 
    to integrate from the current time to `t + t_const`.
    The ICN function uses a second-order scheme equivalent to the iterative Crank-Nicholson algorithm.
    The substep size `delta_t` is controlled by the Courant number to ensure stability.
    Parameters:
    -----------
    maxwell : object
        An object representing the Maxwell simulation environment, which includes:
        - `delta`: The grid spacing (common for all dimensions).
        - `t`: The current time of the simulation.
    fields : list of numpy.ndarray
        A list containing the current field values to be updated, in the following order:
        `[E_x, E_y, E_z, A_x, A_y, A_z, phi]`.
        Each field is a 3D array of shape `(nx, ny, nz)`.
    courant : float
        Courant number to control the substep size `delta_t` for stability.
    t_const : float
        The total time increment over which the simulation should be integrated.
    Returns:
    --------
    tuple (float, list of numpy.ndarray)
        Returns the updated simulation time and the updated fields after the integration.
        The updated fields are in the same order and shapes as `fields`.
    '''
    current_time = maxwell.t
    current_fields = list(fields)
    delta = maxwell.delta
    
    # Calculate maximum stable substep size using 3D CFL condition
    delta_t_max = courant * delta / np.sqrt(3) if courant != 0 else t_const
    
    # Ensure at least one substep to avoid division by zero
    n_steps = max(1, int(np.ceil(t_const / delta_t_max)))
    delta_t = t_const / n_steps
    
    # Perform each substep using the Crank-Nicholson step function
    for _ in range(n_steps):
        current_fields = crank_nicholson_step(maxwell, current_fields, delta_t)
        current_time += delta_t
    
    return current_time, current_fields
