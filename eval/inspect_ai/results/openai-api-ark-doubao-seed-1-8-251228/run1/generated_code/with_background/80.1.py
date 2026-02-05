import os
import math
import time
import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
import pickle
from scipy.constants import  Avogadro



def dist(r1, r2, L):
    '''Calculate the minimum image distance between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    float: The minimum image distance between the two atoms.
    '''
    # Calculate the coordinate differences
    dr = np.array(r1) - np.array(r2)
    # Apply minimum image convention to each component
    dr = dr - L * np.round(dr / L)
    # Compute Euclidean distance of the adjusted difference vector
    distance = np.linalg.norm(dr)
    return distance
