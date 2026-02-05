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
    delta_l = abs(l1 - l2)
    sum_l = l1 + l2
    term1 = np.exp(-q * d * delta_l)
    term2 = alpha * np.exp(-q * d * sum_l)
    form_factor = term1 + term2
    return form_factor
