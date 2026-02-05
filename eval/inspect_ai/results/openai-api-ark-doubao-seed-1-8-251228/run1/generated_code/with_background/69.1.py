import numpy as np



def f_V(q, d, bg_eps, l1, l2):
    '''Write down the form factor f(q;l1,l2)
    Input
    q, in-plane momentum, float in the unit of inverse angstrom
    d, layer spacing, float in the unit of angstrom
    bg_eps: LEG dielectric constant, float, dimensionless
    l1,l2: layer number where z = l*d, integer
    Output
    form_factor: form factor, float
    '''
    alpha = (bg_eps - 1) / (bg_eps + 1)
    distance_direct = d * abs(l1 - l2)
    distance_image = d * (l1 + l2)
    term_direct = np.exp(-q * distance_direct)
    term_image = alpha * np.exp(-q * distance_image)
    form_factor = term_direct + term_image
    return form_factor
