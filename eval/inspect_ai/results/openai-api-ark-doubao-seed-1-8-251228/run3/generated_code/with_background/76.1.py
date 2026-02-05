import numpy as np
import random
from collections import Counter



def load_motif_from_df(data):
    '''Input:
    PWM matrix with keys 'A', 'C', 'G', 'T'
    Output:
    mat: (number of row of PWM matrix, 4) integer array, each row is a probability distribution
    '''
    # Extract the PWM values in the order of 'A', 'C', 'G', 'T'
    pwm_array = data[['A', 'C', 'G', 'T']].values
    # Add 1 to each entry to avoid log divergence
    smoothed_array = pwm_array + 1
    # Calculate row sums for L1 normalization
    row_sums = smoothed_array.sum(axis=1, keepdims=True)
    # Normalize each row to form a probability distribution
    normalized_mat = smoothed_array / row_sums
    return normalized_mat
