import numpy as np

def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    r_np = np.asarray(r)
    coord = np.mod(r_np, L)
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
        '''Compute the Lennard-Jones Potential between a pair of atoms at distance r_ij.'''
        if r_ij < r_c:
            sigma_over_r = sigma / r_ij
            return 4 * epsilon * (sigma_over_r ** 12 - sigma_over_r ** 6)
        else:
            return 0.0
    
    r_np = np.asarray(r)
    pos_np = np.asarray(pos)
    total_energy = 0.0
    
    for pos_j in pos_np:
        # Compute relative position vector
        dr = pos_j - r_np
        # Apply minimum image convention to get shortest periodic distance
        dr = np.mod(dr + L / 2, L) - L / 2
        # Calculate Euclidean distance between particles
        r_ij = np.linalg.norm(dr)
        # Accumulate pairwise interaction energy
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
    # Generate random position for the test particle within the cubic box [0, L)
    test_r = np.random.uniform(0.0, L, size=3)
    
    # Calculate the total insertion energy using the E_i function
    U_insert = E_i(test_r, pos, sigma, epsilon, L, r_c)
    
    # Define beta as 1/(k_B*T), assuming k_B=1 (common in reduced unit simulations)
    beta = 1.0 / T
    
    # Compute the Boltzmann factor
    Boltz = np.exp(-beta * U_insert)
    
    return Boltz


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
    # Calculate box length
    L = (N / rho) ** (1/3)
    
    # Determine smallest integer m where m^3 >= N
    m = 1
    while m ** 3 < N:
        m += 1
    
    # Calculate lattice spacing
    spacing = L / m
    
    # Generate grid indices
    indices = np.indices((m, m, m)).reshape(3, -1).T
    
    # Compute particle positions
    positions = indices * spacing
    
    # Take only the first N positions
    positions = positions[:N]
    
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
        number of accepted moves, and acceptance ratio.'''

    def metropolis_step(positions, U_total, n_accp):
        beta = 1.0 / T
        n_particles = positions.shape[0]
        # Select random particle index
        i = np.random.randint(n_particles)
        r_old = positions[i].copy()
        
        # Create array of positions excluding the selected particle
        positions_without_i = np.concatenate([positions[:i], positions[i+1:]])
        
        # Calculate energy of particle in current position
        E_old = E_i(r_old, positions_without_i, sigma, epsilon, L, r_c)
        
        # Propose new position with random displacement
        displacement = np.random.uniform(-move_magnitude, move_magnitude, size=3)
        r_new = r_old + displacement
        r_new = wrap(r_new, L)  # Apply periodic boundary conditions
        
        # Calculate energy of particle in proposed position
        E_new = E_i(r_new, positions_without_i, sigma, epsilon, L, r_c)
        delta_U = E_new - E_old
        
        # Metropolis acceptance criterion
        if delta_U < 0.0 or np.random.uniform() < np.exp(-beta * delta_U):
            positions[i] = r_new
            U_total += delta_U
            n_accp += 1
        
        return U_total, n_accp

    # Initialize system of particles
    positions, L = init_system(N, rho)
    n_accp = 0  # Counter for accepted moves

    # Calculate initial total potential energy
    sum_E = 0.0
    for i in range(N):
        positions_without_i = np.concatenate([positions[:i], positions[i+1:]])
        sum_E += E_i(positions[i], positions_without_i, sigma, epsilon, L, r_c)
    U_total = 0.5 * sum_E  # Correct for double-counting pairs

    # Equilibration phase: run without data collection
    for _ in range(n_eq):
        U_total, n_accp = metropolis_step(positions, U_total, n_accp)

    # Production phase: collect energy data and perform Widom insertions
    E = np.zeros(n_prod)
    boltzmann_factors = []
    
    for t in range(n_prod):
        U_total, n_accp = metropolis_step(positions, U_total, n_accp)
        # Record total system energy after the step
        E[t] = U_total
        
        # Perform Widom insertion at specified frequency
        if (t + 1) % insertion_freq == 0:
            boltz = Widom_insertion(positions, sigma, epsilon, L, r_c, T)
            boltzmann_factors.append(boltz)

    # Calculate excess chemical potential from collected Boltzmann factors
    if boltzmann_factors:
        average_boltz = np.mean(boltzmann_factors)
        if average_boltz > 0.0:
            ecp = -T * np.log(average_boltz)
        else:
            ecp = np.nan  # Handle non-positive average (numerical edge case)
    else:
        ecp = np.nan  # No insertions performed

    # Calculate acceptance ratio
    total_steps = n_eq + n_prod
    accp_ratio = n_accp / total_steps if total_steps > 0 else 0.0

    return E, ecp, n_accp, accp_ratio
