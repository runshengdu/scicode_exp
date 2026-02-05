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
    direct_term = np.exp(-q * d * np.abs(l1 - l2))
    image_term = alpha * np.exp(-q * d * (l1 + l2))
    form_factor = direct_term + image_term
    return form_factor


print('Hello, World!')


print('Hello, World!')


print('Hello, World!')


print('Hello, World!')


print('Hello, World!')


print('Hello, World!')



print('Hello, World!')
