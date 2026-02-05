import numpy as np
from cmath import exp
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.linalg import expm



def rotation_matrices(axis, theta):
    '''Create rotation matrices Rx, Ry, and Rz with the given angle theta.
    Inputs:
    axis : int
        The rotation axis. 1 = x, 2 = y, 3 = z.
    theta : float
        The rotation angle.
    Output:
    R : matrix of shape(2, 2)
        The rotation matrix.
    '''
    theta_half = theta / 2
    if axis == 1:
        cos_h = np.cos(theta_half)
        sin_h = np.sin(theta_half)
        R = np.array([[cos_h, -1j * sin_h],
                      [-1j * sin_h, cos_h]])
    elif axis == 2:
        cos_h = np.cos(theta_half)
        sin_h = np.sin(theta_half)
        R = np.array([[cos_h, -sin_h],
                      [sin_h, cos_h]])
    elif axis == 3:
        R = np.array([[exp(-1j * theta_half), 0],
                      [0, exp(1j * theta_half)]])
    else:
        raise ValueError("Axis must be 1 (x), 2 (y), or 3 (z)")
    return R
