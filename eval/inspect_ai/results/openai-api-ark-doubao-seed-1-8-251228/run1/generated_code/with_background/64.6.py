import numpy as np
import itertools

def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    coord = np.mod(r, L)
    return coord


def dist(r1, r2, L):
    '''Calculate the minimum image distance between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    float: The minimum image distance between the two atoms.
    '''
    r1_np = np.array(r1)
    r2_np = np.array(r2)
    dr = r1_np - r2_np
    # Apply minimum image convention to the difference vector
    dr = np.mod(dr + L / 2, L) - L / 2
    # Calculate Euclidean distance of the adjusted difference vector
    distance = np.linalg.norm(dr)
    return distance


def E_ij(r, sigma, epsilon):
    '''Calculate the Lennard-Jones potential energy between two particles.
    Parameters:
    r : float
        The distance between the two particles.
    sigma : float
        The distance at which the potential minimum occurs
    epsilon : float
        The depth of the potential well
    Returns:
    float
        The potential energy between the two particles at distance r.
    '''
    sr = sigma / r
    sr12 = sr ** 12
    sr6 = sr ** 6
    E_lj = 4 * epsilon * (sr12 - sr6)
    return E_lj


def E_i(r, positions, L, sigma, epsilon):
    '''Calculate the total Lennard-Jones potential energy of a particle with other particles in a periodic system.
    Parameters:
    r : array_like
        The (x, y, z) coordinates of the target particle.
    positions : array_like
        An array of (x, y, z) coordinates for each of the other particles in the system.
    L : float
        The length of the side of the cubic box
    sigma : float
        The distance at which the potential minimum occurs
    epsilon : float
        The depth of the potential well
    Returns:
    float
        The total Lennard-Jones potential energy of the particle due to its interactions with other particles.
    '''
    total_energy = 0.0
    for pos in positions:
        d = dist(r, pos, L)
        total_energy += E_ij(d, sigma, epsilon)
    return total_energy


def E_system(positions, L, sigma, epsilon):
    '''Calculate the total Lennard-Jones potential energy of a particle with other particles in a periodic system.
    Parameters:
    positions : array_like
        An array of (x, y, z) coordinates for each of the other particles in the system.
    L : float
        The length of the side of the cubic box
    sigma : float
        The distance at which the potential minimum occurs
    epsilon : float
        The depth of the potential well
    Returns:
    float
        The total Lennard-Jones potential
    '''
    total_E = 0.0
    for r1, r2 in itertools.combinations(positions, 2):
        d = dist(r1, r2, L)
        total_E += E_ij(d, sigma, epsilon)
    return total_E



def GCMC(initial_positions, L, T, mu, sigma, epsilon, mass, num_steps, prob_insertion, prob_deletion, disp_size):
    '''Perform a Grand Canonical Monte Carlo (GCMC) simulation to model particle insertions,
    deletions, and displacements within a periodic system, maintaining equilibrium based on
    the chemical potential and the system's energy states.
    Parameters:
    initial_positions : array_like
        Initial positions of particles within the simulation box.
    L : float
        The length of the side of the cubic box.
    T : float
        Temperature of the system.
    mu : float
        Chemical potential used to determine the probability of insertion and deletion.
    sigma : float
        The distance at which the potential minimum occurs
    epsilon : float
        The depth of the potential well
    mass : float
        Mass of a single particle.
    num_steps : int
        Number of steps to perform in the simulation.
    prob_insertion : float
        Probability of attempting a particle insertion.
    prob_deletion : float
        Probability of attempting a particle deletion.
    disp_size : float
        Size factor for the displacement operation.
    Returns:
    - `Energy_Trace`: Array of the total potential energy of the system at each simulation step, tracking energy changes due to particle interactions and movements (float)
    - `Num_particle_Trace`: Array recording the number of particles in the system at each simulation step, used to observe the effects of particle insertions and deletions (float).
    - `Trail_move_counts_tracker`:
    Dictionary with keys 'Insertion', 'Deletion', and 'Move', each mapped to a two-element array:
        - The first element counts the number of attempts for that move type.
        - The second element counts the successful attempts. This tracker is essential for assessing acceptance rates and tuning the simulation parameters.
    - Lambda: float, Thermal de Broglie Wavelength
    '''
    # Physical constants
    h = 6.62607015e-34  # Planck's constant in J·s
    k_B = 1.380649e-23   # Boltzmann constant in J/K
    
    # Compute thermal de Broglie wavelength
    beta = 1.0 / (k_B * T)
    Lambda = np.sqrt((h**2 * beta) / (2 * np.pi * mass))
    
    # Volume of the cubic box
    V = L ** 3
    
    # Initialize current positions, wrapped to [0, L) using wrap function
    initial_positions_np = np.array(initial_positions)
    current_pos = np.array([wrap(r, L) for r in initial_positions_np])
    
    # Initialize current total energy using E_system function
    current_energy = E_system(current_pos, L, sigma, epsilon)
    
    # Initialize traces to record simulation state at each step
    Energy_Trace = np.zeros(num_steps, dtype=np.float64)
    Num_particle_Trace = np.zeros(num_steps, dtype=np.float64)
    
    # Initialize move attempt/success tracker
    Trail_move_counts_tracker = {
        'Insertion': np.array([0, 0], dtype=np.int64),
        'Deletion': np.array([0, 0], dtype=np.int64),
        'Move': np.array([0, 0], dtype=np.int64)
    }
    
    # Main simulation loop
    for step in range(num_steps):
        # Randomly select move type based on given probabilities
        u = np.random.rand()
        
        if u < prob_insertion:
            # Handle Insertion move
            Trail_move_counts_tracker['Insertion'][0] += 1
            
            # Generate new particle position within box bounds
            new_pos = wrap(np.random.uniform(0, L, size=3), L)
            
            # Calculate energy change from inserting new particle
            delta_E = E_i(new_pos, current_pos, L, sigma, epsilon)
            
            # Compute insertion acceptance probability
            current_N = current_pos.shape[0]
            pre_factor = V / (Lambda**3 * (current_N + 1))
            exponent = beta * (mu - delta_E)
            chi_insert = np.minimum(pre_factor * np.exp(exponent), 1.0)
            
            # Acceptance check
            if np.random.rand() < chi_insert:
                current_pos = np.concatenate([current_pos, new_pos.reshape(1, 3)], axis=0)
                current_energy += delta_E
                Trail_move_counts_tracker['Insertion'][1] += 1
        
        elif u < prob_insertion + prob_deletion:
            # Handle Deletion move
            Trail_move_counts_tracker['Deletion'][0] += 1
            
            current_N = current_pos.shape[0]
            if current_N == 0:
                # Cannot delete from empty system, skip
                continue
            
            # Select random particle to remove
            idx = np.random.randint(current_N)
            del_pos = current_pos[idx]
            pos_without_del = np.delete(current_pos, idx, axis=0)
            
            # Calculate energy change from removing the particle
            sum_interactions = E_i(del_pos, pos_without_del, L, sigma, epsilon)
            
            # Compute deletion acceptance probability
            pre_factor = (current_N * Lambda**3) / V
            exponent = beta * (sum_interactions - mu)
            chi_delete = np.minimum(pre_factor * np.exp(exponent), 1.0)
            
            # Acceptance check
            if np.random.rand() < chi_delete:
                current_pos = pos_without_del
                current_energy -= sum_interactions
                Trail_move_counts_tracker['Deletion'][1] += 1
        
        else:
            # Handle Displacement (Move)
            Trail_move_counts_tracker['Move'][0] += 1
            
            current_N = current_pos.shape[0]
            if current_N == 0:
                # Cannot move particles in empty system, skip
                continue
            
            # Select random particle to displace
            idx = np.random.randint(current_N)
            old_pos = current_pos[idx]
            
            # Generate new position with wrapped periodic boundaries
            delta_r = np.random.uniform(-disp_size * L / 2, disp_size * L / 2, size=3)
            new_pos_unwrapped = old_pos + delta_r
            new_pos = wrap(new_pos_unwrapped, L)
            
            # Calculate energy change from displacement
            pos_without_particle = np.delete(current_pos, idx, axis=0)
            E_old = E_i(old_pos, pos_without_particle, L, sigma, epsilon)
            E_new = E_i(new_pos, pos_without_particle, L, sigma, epsilon)
            delta_E = E_new - E_old
            
            # Compute displacement acceptance probability (Metropolis criterion)
            chi_move = np.minimum(np.exp(-beta * delta_E), 1.0)
            
            # Acceptance check
            if np.random.rand() < chi_move:
                current_pos[idx] = new_pos
                current_energy += delta_E
                Trail_move_counts_tracker['Move'][1] += 1
        
        # Record current system state
        Energy_Trace[step] = current_energy
        Num_particle_Trace[step] = current_pos.shape[0]
    
    return Energy_Trace, Num_particle_Trace, Trail_move_counts_tracker, Lambda
