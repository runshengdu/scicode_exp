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
