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
    dx = r1[0] - r2[0]
    dy = r1[1] - r2[1]
    dz = r1[2] - r2[2]
    
    # Apply minimum image correction to each component
    dx = dx - L * round(dx / L)
    dy = dy - L * round(dy / L)
    dz = dz - L * round(dz / L)
    
    # Calculate Euclidean distance
    distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    
    return distance
