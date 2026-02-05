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
    
    # Initialize derivative arrays with the same shape as the input field
    deriv_x = np.zeros_like(fct)
    deriv_y = np.zeros_like(fct)
    deriv_z = np.zeros_like(fct)
    
    # Compute ∂f/∂x
    # Interior points using central second-order difference
    deriv_x[1:-1, :, :] = (fct[2:, :, :] - fct[:-2, :, :]) / (2 * delta)
    # Left boundary (x=0) using forward second-order difference
    deriv_x[0, :, :] = (-3 * fct[0, :, :] + 4 * fct[1, :, :] - fct[2, :, :]) / (2 * delta)
    # Right boundary (x=nx-1) using backward second-order difference
    deriv_x[-1, :, :] = (3 * fct[-1, :, :] - 4 * fct[-2, :, :] + fct[-3, :, :]) / (2 * delta)
    
    # Compute ∂f/∂y
    # Interior points using central second-order difference
    deriv_y[:, 1:-1, :] = (fct[:, 2:, :] - fct[:, :-2, :]) / (2 * delta)
    # Bottom boundary (y=0) using forward second-order difference
    deriv_y[:, 0, :] = (-3 * fct[:, 0, :] + 4 * fct[:, 1, :] - fct[:, 2, :]) / (2 * delta)
    # Top boundary (y=ny-1) using backward second-order difference
    deriv_y[:, -1, :] = (3 * fct[:, -1, :] - 4 * fct[:, -2, :] + fct[:, -3, :]) / (2 * delta)
    
    # Compute ∂f/∂z
    # Interior points using central second-order difference
    deriv_z[:, :, 1:-1] = (fct[:, :, 2:] - fct[:, :, :-2]) / (2 * delta)
    # Front boundary (z=0) using forward second-order difference
    deriv_z[:, :, 0] = (-3 * fct[:, :, 0] + 4 * fct[:, :, 1] - fct[:, :, 2]) / (2 * delta)
    # Back boundary (z=nz-1) using backward second-order difference
    deriv_z[:, :, -1] = (3 * fct[:, :, -1] - 4 * fct[:, :, -2] + fct[:, :, -3]) / (2 * delta)
    
    return deriv_x, deriv_y, deriv_z


def laplace(fct, delta):
    '''Computes the Laplacian of a scalar field in the interior of a 3D grid using second-order finite differences.
    This function calculates the Laplacian of a scalar field on a structured 3D grid using a central finite difference
    scheme. The output boundary values are set to zero to ensure the Laplacian is only calculated for the interior grid points.
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
    lap = np.zeros_like(fct)
    
    # Compute Laplacian for interior points (not on any boundary)
    lap[1:-1, 1:-1, 1:-1] = (
        # Second derivative in x-direction
        (fct[2:, 1:-1, 1:-1] - 2 * fct[1:-1, 1:-1, 1:-1] + fct[:-2, 1:-1, 1:-1]) +
        # Second derivative in y-direction
        (fct[1:-1, 2:, 1:-1] - 2 * fct[1:-1, 1:-1, 1:-1] + fct[1:-1, :-2, 1:-1]) +
        # Second derivative in z-direction
        (fct[1:-1, 1:-1, 2:] - 2 * fct[1:-1, 1:-1, 1:-1] + fct[1:-1, 1:-1, :-2])
    ) / (delta ** 2)
    
    return lap


def gradient(fct, delta):
    '''Computes the gradient of a scalar field in the interior of a 3D grid using second-order finite differences.
    This function calculates the gradient of a scalar field on a structured 3D grid using a central finite difference
    scheme. The output boundary values are set to zero to ensure the gradient is only calculated for the interior grid points.
    Parameters:
    -----------
    fct : numpy.ndarray
        A 3D array representing the scalar field values on the grid. Shape: (nx, ny, nz).
    delta : float
        The grid spacing or step size in all spatial directions.
    Returns:
    --------
    grad_x : numpy.ndarray
        The partial derivative of the field with respect to the x direction (∂f/∂x). 
        Shape: (nx, ny, nz). Boundary values are zeroed out.
    grad_y : numpy.ndarray
        The partial derivative of the field with respect to the y direction (∂f/∂y). 
        Shape: (nx, ny, nz). Boundary values are zeroed out.
    grad_z : numpy.ndarray
        The partial derivative of the field with respect to the z direction (∂f/∂z). 
        Shape: (nx, ny, nz). Boundary values are zeroed out.
    '''
    nx, ny, nz = fct.shape
    
    # Initialize gradient arrays with zeros
    grad_x = np.zeros_like(fct)
    grad_y = np.zeros_like(fct)
    grad_z = np.zeros_like(fct)
    
    # Compute ∂f/∂x for interior x points using central second-order difference
    grad_x[1:-1, :, :] = (fct[2:, :, :] - fct[:-2, :, :]) / (2 * delta)
    
    # Compute ∂f/∂y for interior y points using central second-order difference
    grad_y[:, 1:-1, :] = (fct[:, 2:, :] - fct[:, :-2, :]) / (2 * delta)
    
    # Compute ∂f/∂z for interior z points using central second-order difference
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
    div = np.zeros_like(v_x)
    
    # Compute divergence for interior grid points using central second-order differences
    div[1:-1, 1:-1, 1:-1] = (
        # Partial derivative of x-component with respect to x
        (v_x[2:, 1:-1, 1:-1] - v_x[:-2, 1:-1, 1:-1]) +
        # Partial derivative of y-component with respect to y
        (v_y[1:-1, 2:, 1:-1] - v_y[1:-1, :-2, 1:-1]) +
        # Partial derivative of z-component with respect to z
        (v_z[1:-1, 1:-1, 2:] - v_z[1:-1, 1:-1, :-2])
    ) / (2 * delta)
    
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
    
    # Initialize gradient of divergence arrays with zeros
    grad_div_x = np.zeros_like(A_x)
    grad_div_y = np.zeros_like(A_y)
    grad_div_z = np.zeros_like(A_z)
    
    # Compute x-component of gradient of divergence (∂ₓ(∇·A)) in interior
    grad_div_x[1:-1, 1:-1, 1:-1] = (
        4 * (A_x[2:, 1:-1, 1:-1] - 2 * A_x[1:-1, 1:-1, 1:-1] + A_x[:-2, 1:-1, 1:-1]) +
        (A_y[2:, 2:, 1:-1] - A_y[:-2, 2:, 1:-1] - A_y[2:, :-2, 1:-1] + A_y[:-2, :-2, 1:-1]) +
        (A_z[2:, 1:-1, 2:] - A_z[:-2, 1:-1, 2:] - A_z[2:, 1:-1, :-2] + A_z[:-2, 1:-1, :-2])
    ) / (4 * delta ** 2)
    
    # Compute y-component of gradient of divergence (∂ᵧ(∇·A)) in interior
    grad_div_y[1:-1, 1:-1, 1:-1] = (
        (A_x[2:, 2:, 1:-1] - A_x[:-2, 2:, 1:-1] - A_x[2:, :-2, 1:-1] + A_x[:-2, :-2, 1:-1]) +
        4 * (A_y[1:-1, 2:, 1:-1] - 2 * A_y[1:-1, 1:-1, 1:-1] + A_y[1:-1, :-2, 1:-1]) +
        (A_z[1:-1, 2:, 2:] - A_z[1:-1, :-2, 2:] - A_z[1:-1, 2:, :-2] + A_z[1:-1, :-2, :-2])
    ) / (4 * delta ** 2)
    
    # Compute z-component of gradient of divergence (∂z(∇·A)) in interior
    grad_div_z[1:-1, 1:-1, 1:-1] = (
        (A_x[2:, 1:-1, 2:] - A_x[:-2, 1:-1, 2:] - A_x[2:, 1:-1, :-2] + A_x[:-2, 1:-1, :-2]) +
        (A_y[1:-1, 2:, 2:] - A_y[1:-1, :-2, 2:] - A_y[1:-1, 2:, :-2] + A_y[1:-1, :-2, :-2]) +
        4 * (A_z[1:-1, 1:-1, 2:] - 2 * A_z[1:-1, 1:-1, 1:-1] + A_z[1:-1, 1:-1, :-2])
    ) / (4 * delta ** 2)
    
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
    # Case 1: Only x is negative (i=0, j>=1, k>=1)
    f_dot[0, 1:, 1:] = x_sym * f_dot[1, 1:, 1:]
    
    # Case 2: Only y is negative (j=0, i>=1, k>=1)
    f_dot[1:, 0, 1:] = y_sym * f_dot[1:, 1, 1:]
    
    # Case 3: Only z is negative (k=0, i>=1, j>=1)
    f_dot[1:, 1:, 0] = z_sym * f_dot[1:, 1:, 1]
    
    # Case 4: x and y are negative (i=0, j=0, k>=1)
    f_dot[0, 0, 1:] = x_sym * y_sym * f_dot[1, 1, 1:]
    
    # Case 5: x and z are negative (i=0, k=0, j>=1)
    f_dot[0, 1:, 0] = x_sym * z_sym * f_dot[1, 1:, 1]
    
    # Case 6: y and z are negative (j=0, k=0, i>=1)
    f_dot[1:, 0, 0] = y_sym * z_sym * f_dot[1:, 1, 1]
    
    # Case 7: x, y, z all negative (i=0, j=0, k=0)
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
    nx, ny, nz = f.shape
    delta = maxwell.delta
    x = maxwell.x
    y = maxwell.y
    z = maxwell.z
    r = maxwell.r

    # Compute partial derivatives ∂x f
    dx_f = np.zeros_like(f)
    # Interior x points
    dx_f[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2 * delta)
    # Inner x boundary (i=0) using forward second-order difference
    dx_f[0, :, :] = (-3 * f[0, :, :] + 4 * f[1, :, :] - f[2, :, :]) / (2 * delta)
    # Outer x boundary (i=nx-1) using backward second-order difference
    dx_f[-1, :, :] = (3 * f[-1, :, :] - 4 * f[-2, :, :] + f[-3, :, :]) / (2 * delta)

    # Compute partial derivatives ∂y f
    dy_f = np.zeros_like(f)
    # Interior y points
    dy_f[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2 * delta)
    # Inner y boundary (j=0) using forward second-order difference
    dy_f[:, 0, :] = (-3 * f[:, 0, :] + 4 * f[:, 1, :] - f[:, 2, :]) / (2 * delta)
    # Outer y boundary (j=ny-1) using backward second-order difference
    dy_f[:, -1, :] = (3 * f[:, -1, :] - 4 * f[:, -2, :] + f[:, -3, :]) / (2 * delta)

    # Compute partial derivatives ∂z f
    dz_f = np.zeros_like(f)
    # Interior z points
    dz_f[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * delta)
    # Inner z boundary (k=0) using forward second-order difference
    dz_f[:, :, 0] = (-3 * f[:, :, 0] + 4 * f[:, :, 1] - f[:, :, 2]) / (2 * delta)
    # Outer z boundary (k=nz-1) using backward second-order difference
    dz_f[:, :, -1] = (3 * f[:, :, -1] - 4 * f[:, :, -2] + f[:, :, -3]) / (2 * delta)

    # Compute the right-hand side of the outgoing wave condition
    rhs = (-f - (x * dx_f + y * dy_f + z * dz_f)) / r

    # Create mask for outer boundary points (i=nx-1, j=ny-1, or k=nz-1)
    i_indices, j_indices, k_indices = np.indices(f.shape)
    outer_mask = (i_indices == nx-1) | (j_indices == ny-1) | (k_indices == nz-1)

    # Apply outgoing wave condition to outer boundary points
    f_dot[outer_mask] = rhs[outer_mask]

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
        A tuple containing the current field values in the following order:
        `(E_x, E_y, E_z, A_x, A_y, A_z, phi)`.
        Each component is a 3D array of shape `(nx, ny, nz)`.
    Returns:
    --------
    tuple of numpy.ndarray
        A tuple containing the time derivatives of the fields in the following order:
        `(E_x_dot, E_y_dot, E_z_dot, A_x_dot, A_y_dot, A_z_dot, phi_dot)`.
        Each component is a 3D array of shape `(nx, ny, nz)`.
    '''
    # Extract field components from input tuple
    E_x, E_y, E_z, A_x, A_y, A_z, phi = fields
    delta = maxwell.delta

    # ------------------------------
    # Compute E field time derivatives
    # ------------------------------
    # Calculate Laplacian of each vector potential component
    lap_Ax = laplace(A_x, delta)
    lap_Ay = laplace(A_y, delta)
    lap_Az = laplace(A_z, delta)
    
    # Calculate gradient of divergence of the vector potential
    gdiv_x, gdiv_y, gdiv_z = grad_div(A_x, A_y, A_z, delta)
    
    # Initial E_dot calculation using Lorentz gauge equation
    E_x_dot = -lap_Ax + gdiv_x
    E_y_dot = -lap_Ay + gdiv_y
    E_z_dot = -lap_Az + gdiv_z
    
    # Apply inner boundary symmetry conditions
    # E_x: antisymmetric across x=0, symmetric across y=0/z=0
    E_x_dot = symmetry(E_x_dot, x_sym=-1, y_sym=1, z_sym=1)
    # E_y: antisymmetric across y=0, symmetric across x=0/z=0
    E_y_dot = symmetry(E_y_dot, x_sym=1, y_sym=-1, z_sym=1)
    # E_z: antisymmetric across z=0, symmetric across x=0/y=0
    E_z_dot = symmetry(E_z_dot, x_sym=1, y_sym=1, z_sym=-1)
    
    # Apply outer boundary outgoing wave conditions
    E_x_dot = outgoing_wave(maxwell, E_x_dot, E_x)
    E_y_dot = outgoing_wave(maxwell, E_y_dot, E_y)
    E_z_dot = outgoing_wave(maxwell, E_z_dot, E_z)

    # ------------------------------
    # Compute A field time derivatives
    # ------------------------------
    # Calculate gradient of scalar potential phi
    grad_phi_x, grad_phi_y, grad_phi_z = gradient(phi, delta)
    
    # Initial A_dot calculation using Lorentz gauge equation
    A_x_dot = -E_x - grad_phi_x
    A_y_dot = -E_y - grad_phi_y
    A_z_dot = -E_z - grad_phi_z
    
    # Apply inner boundary symmetry conditions (A is polar vector, same as E)
    A_x_dot = symmetry(A_x_dot, x_sym=-1, y_sym=1, z_sym=1)
    A_y_dot = symmetry(A_y_dot, x_sym=1, y_sym=-1, z_sym=1)
    A_z_dot = symmetry(A_z_dot, x_sym=1, y_sym=1, z_sym=-1)
    
    # Apply outer boundary outgoing wave conditions
    A_x_dot = outgoing_wave(maxwell, A_x_dot, A_x)
    A_y_dot = outgoing_wave(maxwell, A_y_dot, A_y)
    A_z_dot = outgoing_wave(maxwell, A_z_dot, A_z)

    # ------------------------------
    # Compute phi time derivative
    # ------------------------------
    # Calculate divergence of vector potential A
    div_A = divergence(A_x, A_y, A_z, delta)
    
    # Initial phi_dot calculation using Lorentz gauge condition
    phi_dot = -div_A
    
    # Apply inner boundary symmetry conditions (phi is scalar, fully symmetric)
    phi_dot = symmetry(phi_dot, x_sym=1, y_sym=1, z_sym=1)
    
    # Apply outer boundary outgoing wave conditions
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
    # Calculate updated fields by applying scaled time derivative increment
    new_fields = [current_field + factor * dt * derivative 
                  for current_field, derivative in zip(fields, fields_dot)]
    
    return new_fields
