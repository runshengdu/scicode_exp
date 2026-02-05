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



def compute_kld(matrix):
    '''Input:
    (number of row of PWM matrix, 4) array, PWM
    Output:
    Kullback–Leibler divergence (float)
    '''
    # Uniform background distribution probability for each nucleotide
    background_prob = 0.25
    # Calculate element-wise KL divergence terms using natural logarithm
    kl_terms = matrix * np.log(matrix / background_prob)
    # Sum all terms to get the total KL divergence
    total_kld = np.sum(kl_terms)
    return total_kld
