import numpy as np

def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    r_array = np.asarray(r)
    coord = np.mod(r_array, L)
    return coord


def E_i(r, pos, sigma, epsilon, L, r_c):
    '''Calculate the total Lennard-Jones potential energy of a particle with other particles in a periodic system.
    Parameters:
    r : array, the (x, y, z) coordinates of the target particle.
    pos : An array of (x, y, z) coordinates for each of the other particles in the system.
    sigma : float, the distance at which the potential minimum occurs
    epsilon : float, the depth of the potential well
    L : float, the length of the side of the cubic box
    r_c : float, cut-off distance
    Returns:
    float, the total Lennard-Jones potential energy of the particle due to its interactions with other particles.
    '''
    def E_ij(r_ij):
        '''Compute the Lennard-Jones Potential between a pair of atoms.'''
        if r_ij < r_c:
            sigma_over_r = sigma / r_ij
            return 4 * epsilon * (sigma_over_r ** 12 - sigma_over_r ** 6)
        return 0.0
    
    total_energy = 0.0
    r_array = np.asarray(r)
    pos_array = np.asarray(pos)
    
    for pos_j in pos_array:
        # Calculate separation vector between target particle and current particle
        delta = pos_j - r_array
        # Apply minimum image convention to get shortest periodic distance
        delta = delta - L * np.round(delta / L)
        # Compute Euclidean distance between the particles
        r_ij = np.linalg.norm(delta)
        # Accumulate pairwise potential energy
        total_energy += E_ij(r_ij)
    
    return total_energy


def Widom_insertion(pos, sigma, epsilon, L, r_c, T):
    '''Perform the Widom test particle insertion method to calculate the change in chemical potential.
    Parameters:
    pos : ndarray, Array of position vectors of all particles in the system.
    sigma: float, The effective particle diameter 
    epsilon: float, The depth of the potential well
    L: float, The length of each side of the cubic simulation box
    r_c: float, Cut-Off Distance
    T: float, The temperature of the system
    Returns:
    float: Boltzmann factor for the test particle insertion, e^(-beta * energy of insertion).
    '''
    # Generate random position for test particle within the cubic box [0, L)
    r_test = np.random.uniform(0.0, L, size=3)
    
    # Calculate insertion energy using the previously defined E_i function
    E_insertion = E_i(r_test, pos, sigma, epsilon, L, r_c)
    
    # Compute beta (1/(k_B T)), assuming k_B = 1 (common in reduced unit simulations)
    beta = 1.0 / T
    
    # Calculate and return the Boltzmann factor
    boltz_factor = np.exp(-beta * E_insertion)
    return boltz_factor


def init_system(N, rho):
    '''Initialize a system of particles arranged on a cubic grid within a cubic box.
    Args:
    N (int): The number of particles to be placed in the box.
    rho (float): The density of particles within the box, defined as the number of particles per unit volume.
    Returns:
    tuple: A tuple containing:
        - positions(np.ndarray): The array of particle positions in a 3D space.
        - L(float): The length of the side of the cubic box in which the particles are placed.
    '''
    # Calculate the side length of the cubic box
    L = (N / rho) ** (1/3)
    
    # Determine the number of particles along each axis (ceiling of cube root of N)
    m = int(np.ceil(N ** (1/3)))
    
    # Generate evenly spaced coordinates along each axis within [0, L)
    x = np.linspace(0.0, L, m, endpoint=False)
    y = np.linspace(0.0, L, m, endpoint=False)
    z = np.linspace(0.0, L, m, endpoint=False)
    
    # Create 3D grid of positions using Cartesian indexing
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Flatten grid to N x 3 array of particle positions
    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T[:N]
    
    return positions, L



def MC(N, sigma, epsilon, r_c, rho, T, n_eq, n_prod, insertion_freq, move_magnitude):
    '''Perform Monte Carlo simulations using the Metropolis-Hastings algorithm and Widom insertion method to calculate system energies and chemical potential.
    Parameters:
    N (int): The number of particles to be placed in the box.
    sigma, epsilon : float
        Parameters of the Lennard-Jones potential.
    r_c : float
        Cutoff radius beyond which the LJ potential is considered zero.
    rho (float): The density of particles within the box, defined as the number of particles per unit volume.
    T : float
        Temperature of the system.
    n_eq : int
        Number of equilibration steps in the simulation.
    n_prod : int
        Number of production steps in the simulation.
    insertion_freq : int
        Frequency of performing Widom test particle insertions after equilibration.
    move_magnitude : float
        Magnitude of the random displacement in particle movement.
    Returns:
    tuple
        Returns a tuple containing the corrected energy array, extended chemical potential,
        number of accepted moves, and acceptance ratio.
    '''
    # Initialize system
    positions, L = init_system(N, rho)
    
    # Helper function to compute total truncated potential energy
    def total_truncated_energy(pos):
        total = 0.0
        for i in range(N):
            other_pos = np.delete(pos, i, axis=0)
            total += E_i(pos[i], other_pos, sigma, epsilon, L, r_c)
        return total / 2.0  # Divide by 2 to avoid double-counting pairs
    
    # Initial total truncated energy
    current_total_energy = total_truncated_energy(positions)
    
    # Calculate tail correction for potential energy
    sigma6 = sigma ** 6
    sigma12 = sigma6 ** 2
    rc3 = r_c ** 3
    rc9 = rc3 ** 3
    term1 = sigma12 / (9 * rc9)
    term2 = sigma6 / (3 * rc3)
    U_tail = 8 * np.pi * epsilon * N * rho * (term1 - term2)
    
    # Simulation parameters
    n_accp = 0
    beta = 1.0 / T
    
    # Equilibration phase
    for _ in range(n_eq):
        # Select random particle
        i = np.random.randint(N)
        r_old = positions[i].copy()
        
        # Generate random displacement
        displacement = np.random.uniform(-move_magnitude/2, move_magnitude/2, size=3)
        r_new_unwrapped = r_old + displacement
        r_new = wrap(r_new_unwrapped, L)
        
        # Compute energy change for the move
        other_pos = np.delete(positions, i, axis=0)
        E_old = E_i(r_old, other_pos, sigma, epsilon, L, r_c)
        E_new = E_i(r_new, other_pos, sigma, epsilon, L, r_c)
        delta_U = E_new - E_old
        
        # Metropolis acceptance criterion
        if delta_U <= 0.0 or np.random.rand() < np.exp(-beta * delta_U):
            positions[i] = r_new
            current_total_energy += delta_U
            n_accp += 1
    
    # Production phase with data collection
    energy_array = []
    boltz_factors = []
    
    for step in range(n_prod):
        # Perform Metropolis move
        i = np.random.randint(N)
        r_old = positions[i].copy()
        
        displacement = np.random.uniform(-move_magnitude/2, move_magnitude/2, size=3)
        r_new_unwrapped = r_old + displacement
        r_new = wrap(r_new_unwrapped, L)
        
        other_pos = np.delete(positions, i, axis=0)
        E_old = E_i(r_old, other_pos, sigma, epsilon, L, r_c)
        E_new = E_i(r_new, other_pos, sigma, epsilon, L, r_c)
        delta_U = E_new - E_old
        
        if delta_U <= 0.0 or np.random.rand() < np.exp(-beta * delta_U):
            positions[i] = r_new
            current_total_energy += delta_U
            n_accp += 1
        
        # Collect corrected total energy (truncated + tail correction)
        corrected_energy = current_total_energy + U_tail
        energy_array.append(corrected_energy)
        
        # Perform Widom insertion at specified frequency
        if (step + 1) % insertion_freq == 0:
            boltz_factor = Widom_insertion(positions, sigma, epsilon, L, r_c, T)
            boltz_factors.append(boltz_factor)
    
    # Calculate acceptance ratio
    total_steps = n_eq + n_prod
    accp_ratio = n_accp / total_steps if total_steps > 0 else 0.0
    
    # Calculate excess chemical potential
    if not boltz_factors:
        mu_ex = np.nan  # No insertion data available
    else:
        avg_boltz = np.mean(boltz_factors)
        mu_ex = -T * np.log(avg_boltz) if avg_boltz > 0 else np.nan
    
    # Convert energy array to numpy array
    energy_array = np.array(energy_array)
    
    return energy_array, mu_ex, n_accp, accp_ratio
