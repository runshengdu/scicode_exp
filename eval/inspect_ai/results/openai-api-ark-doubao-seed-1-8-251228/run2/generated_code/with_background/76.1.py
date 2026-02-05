import numpy as np
import random
from collections import Counter



def load_motif_from_df(data):
    '''Input:
    PWM matrix with keys 'A', 'C', 'G', 'T'
    Output:
    mat: (number of row of PWM matrix, 4) integer array, each row is a probability distribution
    '''
    nucleotide_order = ['A', 'C', 'G', 'T']
    # Extract the numerical values from the DataFrame in the specified nucleotide order
    pwm_counts = np.array(data[nucleotide_order].values)
    # Add pseudocount of 1 to each entry to avoid log divergence
    pwm_with_pseudocount = pwm_counts + 1
    # Calculate row sums while maintaining 2D shape for broadcasting
    row_sums = pwm_with_pseudocount.sum(axis=1, keepdims=True)
    # Perform L1 normalization to convert each row to a probability distribution
    normalized_mat = pwm_with_pseudocount / row_sums
    return normalized_mat
