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


def compute_kld(matrix):
    '''Input:
    (number of row of PWM matrix, 4) array, PWM
    Output:
    Kullback-Leibler divergence (float)
    '''
    background = 0.25
    # Calculate KL divergence for each position and sum all positions
    kl_per_position = np.sum(matrix * np.log(matrix / background), axis=1)
    total_kld = np.sum(kl_per_position)
    return total_kld

def generate_dna(N: int, PWM: dict) -> tuple:
    """
    Input:
    N (int): Length of the resultant DNA sequence.
    PWM matrix with keys 'A', 'C', 'G', 'T'

    Output:
    tuple: Insertion location (int), DNA sequence (str), DNA reverse complement (str)
    """
    p = random.randint(0, N - 1)
    nucleotide = 'ACGT'
    uni_weights = [0.25, 0.25, 0.25, 0.25]
    dna_string = ''.join(random.choices(nucleotide, uni_weights, k=N))
    spike_mat = load_motif_from_df(PWM)
    spiked_seq = ''.join((random.choices(nucleotide, weights=[PWM[nuc][i] for nuc in nucleotide], k=1)[0] for i in range(len(PWM['A']))))
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    reversed_seq = dna_string[::-1]
    reverse_complement = ''.join((complement[nuc] for nuc in reversed_seq if nuc in complement))
    new_seq = dna_string[:p] + spiked_seq + dna_string[p:]
    new_seq_rc = reverse_complement[:N - p] + spiked_seq + reverse_complement[N - p:]
    return (p, new_seq, new_seq_rc)


def scan_sequence(sequence, matrix, scale, num_runs=100):
    '''Input:
    DNA sequence (str)
    matrix (PWM)
    scale (float) 0<scale<1 , 0.8 should be good, too low might cause false positive
    number of run (int, default = 100)
    Output:
    Detected positions (int)
    '''
    # Process PWM into normalized probability matrix
    normalized_mat = load_motif_from_df(matrix)
    motif_length = normalized_mat.shape[0]
    sequence_length = len(sequence)
    
    # Handle edge cases where motif length is 0 or sequence is shorter than motif
    if motif_length == 0 or sequence_length < motif_length:
        return -1
    
    # Compute expectation value (total KL divergence of PWM from uniform background)
    expectation = compute_kld(normalized_mat)
    threshold = scale * expectation
    
    # Precompute logodds matrix and nucleotide index mapping
    nuc_order = ['A', 'C', 'G', 'T']
    nuc_to_idx = {nuc: idx for idx, nuc in enumerate(nuc_order)}
    logodds_mat = np.log(normalized_mat / 0.25)  # Q=0.25 for uniform background
    
    # Precompute scores for all possible windows
    window_scores = []
    for start in range(sequence_length - motif_length + 1):
        window = sequence[start:start+motif_length]
        score = 0.0
        valid_window = True
        for pos in range(motif_length):
            nucleotide = window[pos]
            if nucleotide not in nuc_to_idx:
                valid_window = False
                break
            score += logodds_mat[pos][nuc_to_idx[nucleotide]]
        window_scores.append(score if valid_window else -np.inf)
    
    # Run scanner num_runs times to collect candidate positions
    counter = Counter()
    for _ in range(num_runs):
        # Collect positions where score exceeds threshold
        candidates = [
            start for start in range(sequence_length - motif_length + 1)
            if window_scores[start] > threshold
        ]
        if candidates:
            selected_pos = random.choice(candidates)
            counter[selected_pos] += 1
    
    # Determine most frequent position or return -1 if no candidates found
    if counter:
        most_common_position = counter.most_common(1)[0][0]
    else:
        most_common_position = -1
    
    return most_common_position
