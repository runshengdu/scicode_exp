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
    # Nucleotide to index mapping consistent with load_motif_from_df's order
    nt_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    motif_length = matrix.shape[0]
    sequence_length = len(sequence)
    
    # Handle case where sequence is shorter than the motif length
    if sequence_length < motif_length:
        return -1
    
    # Calculate expectation value using total KL divergence
    expectation_value = compute_kld(matrix)
    threshold = scale * expectation_value
    
    # Precompute logodds scores for all valid windows
    logodds_scores = []
    for start in range(sequence_length - motif_length + 1):
        window = sequence[start:start+motif_length]
        score = 0.0
        valid_window = True
        for pos in range(motif_length):
            nucleotide = window[pos]
            nt_index = nt_to_idx.get(nucleotide)
            if nt_index is None:
                valid_window = False
                break
            prob = matrix[pos][nt_index]
            # Ensure we don't take log of non-positive values (guaranteed by pseudocounts)
            if prob <= 0:
                score = -np.inf
                break
            score += np.log(prob / 0.25)
        logodds_scores.append(score if valid_window else -np.inf)
    
    # Collect candidate positions where logodds exceeds threshold
    candidates = [i for i, score in enumerate(logodds_scores) if score > threshold]
    
    # Handle case where no valid candidates exist
    if not candidates:
        return -1
    
    # Run scanner num_runs times, selecting random candidate each iteration
    position_counter = Counter()
    for _ in range(num_runs):
        selected_position = random.choice(candidates)
        position_counter[selected_position] += 1
    
    # Retrieve the most frequently detected position
    most_common_position = position_counter.most_common(1)[0][0]
    return most_common_position
