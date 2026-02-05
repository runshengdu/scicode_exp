import numpy as np
from scipy.integrate import simps



def simulate_light_diffraction(n, d, RL, R0, lambda_):
    '''Function to simulate light diffraction through a lens and plot the intensity distribution.
    Inputs:
    n (float): Refractive index of the lens material, e.g., 1.5062 for K9 glass.
    d (float): Center thickness of the lens in millimeters (mm).
    RL (float): Radius of curvature of the lens in millimeters (mm), positive for convex surfaces.
    R0 (float): Radius of the incident light beam in millimeters (mm).
    lambda_ (float): Wavelength of the incident light in millimeters (mm), e.g., 1.064e-3 mm for infrared light.
    Outputs:
    Ie (numpy.ndarray): A 2D array of intensity values of the diffraction pattern where Ie[i][j] is the ith x and jth y.
    '''
    # Step 1: Compute lens focal length using thick lens formula for symmetric biconvex lens
    term1 = 2 / RL
    term2 = ((n - 1) * d) / (n * RL ** 2)
    inv_F = (n - 1) * (term1 - term2)
    F = 1 / inv_F if np.abs(inv_F) > 1e-12 else np.inf  # Avoid division by zero

    # Step 2: Incident Gaussian beam parameters (waist located at the lens, Z1=0)
    k = 2 * np.pi / lambda_
    omega10 = R0
    b1 = 2 * np.pi * omega10 ** 2 / lambda_

    # Step 3: Compute transformed beam parameters (new waist position and radius)
    if F == np.inf:
        Z2 = 0.0
        omega20 = omega10
    else:
        D = F ** 2 + (b1 / 2) ** 2
        Z2 = (F * (b1 / 2) ** 2) / D
        b2 = (F ** 2 * b1) / D
        omega20 = np.sqrt(lambda_ * b2 / (2 * np.pi))

    # Step 4: Discretize initial field on the lens plane
    mr0 = 81
    r0_max = 3 * omega10  # Field is negligible beyond 3*omega10
    r0 = np.linspace(0, r0_max, mr0)
    dr0 = r0_max / (mr0 - 1) if mr0 > 1 else 0.0

    # Initial Gaussian field (normalized amplitude)
    E0 = (1 / omega10) * np.exp(-r0 ** 2 / omega10 ** 2)

    # Apply lens phase shift
    if F == np.inf:
        lens_phase = np.ones_like(r0, dtype=np.complex128)
    else:
        lens_phase = np.exp(-1j * k * r0 ** 2 / (2 * F))
    E_after_lens = E0 * lens_phase

    # Step 5: Discretize observation plane (new waist plane at Z2)
    mr2 = 51
    r_max = 3 * omega20  # Cover the main energy lobe of the focused beam
    r = np.linspace(0, r_max, mr2)

    # Compute radial intensity distribution
    if Z2 < 1e-12:  # Observation plane is lens plane
        I_r = np.abs(np.interp(r, r0, E_after_lens)) ** 2
    else:
        ne2 = 61
        theta0 = np.linspace(0, 2 * np.pi, ne2, endpoint=False)
        cos_theta0 = np.cos(theta0)
        dtheta0 = 2 * np.pi / ne2
        E_r = np.zeros(mr2, dtype=np.complex128)

        for m in range(mr2):
            r_m = r[m]
            integral = 0.0 + 0.0j
            for n in range(mr0):
                r0_n = r0[n]
                # Precompute common exponential terms
                exponent_const = 1j * k * (r0_n ** 2 + r_m ** 2) / (2 * Z2)
                exponent_cos = -1j * k * r_m * r0_n / Z2
                # Vectorized angular integration
                theta_sum = np.sum(np.exp(exponent_const + exponent_cos * cos_theta0)) * dtheta0
                integral += E_after_lens[n] * r0_n * dr0 * theta_sum

            # Apply free-space propagation factor
            propagation_factor = np.exp(1j * k * Z2) / (1j * lambda_ * Z2)
            E_r[m] = propagation_factor * integral

        I_r = np.abs(E_r) ** 2

    # Convert radial intensity to Cartesian 2D array
    x = np.linspace(-r_max, r_max, mr2)
    y = np.linspace(-r_max, r_max, mr2)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)

    # Interpolate radial intensity to Cartesian grid
    Ie = np.interp(R.ravel(), r, I_r).reshape(R.shape)
    # Set intensity outside the beam to zero
    Ie[R > r_max] = 0.0

    return Ie
