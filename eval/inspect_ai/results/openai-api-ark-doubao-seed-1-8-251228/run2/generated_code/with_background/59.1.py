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
        c = np.cos(theta_half)
        s = np.sin(theta_half)
        R = np.array([[c, -1j * s],
                      [-1j * s, c]])
    elif axis == 2:
        c = np.cos(theta_half)
        s = np.sin(theta_half)
        R = np.array([[c, -s],
                      [s, c]])
    elif axis == 3:
        e1 = exp(-1j * theta_half)
        e2 = exp(1j * theta_half)
        R = np.array([[e1, 0],
                      [0, e2]])
    else:
        raise ValueError("Axis must be 1 (x), 2 (y), or 3 (z).")
    return R
