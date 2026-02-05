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
    theta_over_2 = theta / 2
    if axis == 1:
        c = np.cos(theta_over_2)
        s = np.sin(theta_over_2)
        R = np.array([[c, -1j * s],
                      [-1j * s, c]])
    elif axis == 2:
        c = np.cos(theta_over_2)
        s = np.sin(theta_over_2)
        R = np.array([[c, -s],
                      [s, c]])
    elif axis == 3:
        diag1 = exp(-1j * theta_over_2)
        diag2 = exp(1j * theta_over_2)
        R = np.array([[diag1, 0],
                      [0, diag2]])
    else:
        raise ValueError("Axis must be 1 (x), 2 (y), or 3 (z).")
    return R
