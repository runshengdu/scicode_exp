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
    # Compute the coordinate differences
    delta_x = r1[0] - r2[0]
    delta_y = r1[1] - r2[1]
    delta_z = r1[2] - r2[2]
    
    # Apply minimum image correction to each coordinate difference
    delta_x = delta_x - L * round(delta_x / L)
    delta_y = delta_y - L * round(delta_y / L)
    delta_z = delta_z - L * round(delta_z / L)
    
    # Calculate the Euclidean distance
    distance = math.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
    
    return distance
