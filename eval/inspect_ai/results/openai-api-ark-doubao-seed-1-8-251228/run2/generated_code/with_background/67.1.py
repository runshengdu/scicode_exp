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
    alpha = (bg_eps - 1.0) / (bg_eps + 1.0)
    exponent_direct = q * d * abs(l1 - l2)
    term_direct = np.exp(-exponent_direct)
    exponent_image = q * d * (l1 + l2)
    term_image = np.exp(-exponent_image)
    form_factor = term_direct + alpha * term_image
    return form_factor
