import numpy as np



def neighbor_list(site, N):
    '''Return all nearest neighbors of site (i, j).
    Args:
        site (Tuple[int, int]): site indices
        N (int): number of sites along each dimension
    Return:
        list: a list of 2-tuples, [(i_left, j_left), (i_above, j_above), (i_right, j_right), (i_below, j_below)]
    '''
    i, j = site
    left = (i, (j - 1) % N)
    above = ((i - 1) % N, j)
    right = (i, (j + 1) % N)
    below = ((i + 1) % N, j)
    nn_wrap = [left, above, right, below]
    return nn_wrap
