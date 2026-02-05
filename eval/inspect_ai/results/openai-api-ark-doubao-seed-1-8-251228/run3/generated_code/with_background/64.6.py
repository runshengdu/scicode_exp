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
    r_np = np.asarray(r)
    coord = np.mod(r_np, L)
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
    # Convert coordinates to numpy arrays for element-wise operations
    r1_np = np.asarray(r1)
    r2_np = np.asarray(r2)
    
    # Compute the difference vector between the two atoms
    dr = r1_np - r2_np
    
    # Apply minimum image correction to each component of the difference vector
    # This adjusts each component to the range [-L/2, L/2]
    dr -= L * np.round(dr / L)
    
    # Calculate the Euclidean norm of the corrected difference vector
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
    E_lj = 4 * epsilon * (sr ** 12 - sr ** 6)
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
        distance = dist(r, pos, L)
        total_energy += E_ij(distance, sigma, epsilon)
    return total_energy


def E_system(positions, L, sigma, epsilon):
    '''Calculate the total Lennard-Jones potential energy of the whole system in a periodic cubic system.
    Parameters:
    positions : array_like
        An array of (x, y, z) coordinates for all particles in the system.
    L : float
        The length of the side of the cubic box
    sigma : float
        The distance at which the potential minimum occurs
    epsilon : float
        The depth of the potential well
    Returns:
    float
        The total Lennard-Jones potential energy of the system.
    '''
    total_E = 0.0
    positions_np = np.asarray(positions)
    n_particles = len(positions_np)
    
    # Iterate over all unique particle pairs (i, j) where i < j to avoid double-counting
    for i, j in itertools.combinations(range(n_particles), 2):
        r1 = positions_np[i]
        r2 = positions_np[j]
        # Compute minimum image distance between the two particles
        distance = dist(r1, r2, L)
        # Add pairwise Lennard-Jones energy to total system energy
        total_E += E_ij(distance, sigma, epsilon)
    
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
    
    # Initialize current positions and particle count
    current_positions = np.asarray(initial_positions).copy()
    n_particles = len(current_positions)
    
    # Compute initial system energy
    if n_particles == 0:
        current_energy = 0.0
    else:
        current_energy = E_system(current_positions, L, sigma, epsilon)
    
    # Initialize traces for energy and particle count
    Energy_Trace = [current_energy]
    Num_particle_Trace = [n_particles]
    
    # Initialize move tracker
    Trail_move_counts_tracker = {
        'Insertion': [0, 0],
        'Deletion': [0, 0],
        'Move': [0, 0]
    }
    
    # Perform simulation steps
    for _ in range(num_steps):
        # Decide which move to attempt
        u = np.random.rand()
        if u < prob_insertion:
            # Attempt insertion
            Trail_move_counts_tracker['Insertion'][0] += 1
            # Generate random position within the box
            r_new = np.random.uniform(0, L, size=3)
            # Calculate energy change if insertion is accepted
            if n_particles == 0:
                delta_U = 0.0
            else:
                delta_U = E_i(r_new, current_positions, L, sigma, epsilon)
            # Compute acceptance probability
            V = L**3
            accept_prob = (V / (Lambda**3 * (n_particles + 1))) * np.exp(beta * (mu - delta_U))
            accept_prob = min(accept_prob, 1.0)
            # Determine acceptance
            if np.random.rand() < accept_prob:
                # Accept insertion: add new particle to positions
                current_positions = np.vstack([current_positions, r_new.reshape(1, 3)])
                current_energy += delta_U
                n_particles += 1
                Trail_move_counts_tracker['Insertion'][1] += 1
        elif u < prob_insertion + prob_deletion:
            # Attempt deletion
            Trail_move_counts_tracker['Deletion'][0] += 1
            if n_particles > 0:
                # Select random particle to delete
                i = np.random.randint(n_particles)
                r_del = current_positions[i]
                # Get positions excluding the selected particle
                positions_without_i = current_positions[np.arange(n_particles) != i]
                # Calculate energy change (U(N-1) - U(N))
                E_deleted = E_i(r_del, positions_without_i, L, sigma, epsilon)
                delta_U = -E_deleted
                # Compute acceptance probability
                V = L**3
                accept_prob = (Lambda**3 * n_particles / V) * np.exp(beta * (E_deleted - mu))
                accept_prob = min(accept_prob, 1.0)
                # Determine acceptance
                if np.random.rand() < accept_prob:
                    # Accept deletion: remove particle from positions
                    current_positions = np.delete(current_positions, i, axis=0)
                    current_energy += delta_U
                    n_particles -= 1
                    Trail_move_counts_tracker['Deletion'][1] += 1
        else:
            # Attempt displacement (Move)
            Trail_move_counts_tracker['Move'][0] += 1
            if n_particles > 0:
                # Select random particle to displace
                i = np.random.randint(n_particles)
                r_old = current_positions[i]
                # Generate displacement vector
                delta = np.random.uniform(-disp_size, disp_size, size=3)
                r_new = r_old + delta
                # Apply periodic boundary conditions
                r_new_wrapped = wrap(r_new, L)
                # Get positions excluding the selected particle
                positions_without_i = current_positions[np.arange(n_particles) != i]
                # Calculate energy change
                E_old = E_i(r_old, positions_without_i, L, sigma, epsilon)
                E_new = E_i(r_new_wrapped, positions_without_i, L, sigma, epsilon)
                delta_U = E_new - E_old
                # Compute acceptance probability
                accept_prob = np.exp(-beta * delta_U)
                accept_prob = min(accept_prob, 1.0)
                # Determine acceptance
                if np.random.rand() < accept_prob:
                    # Accept displacement: update particle position
                    current_positions[i] = r_new_wrapped
                    current_energy += delta_U
                    Trail_move_counts_tracker['Move'][1] += 1
        # Record current state after step
        Energy_Trace.append(current_energy)
        Num_particle_Trace.append(n_particles)
    
    # Convert traces to numpy arrays
    Energy_Trace = np.array(Energy_Trace)
    Num_particle_Trace = np.array(Num_particle_Trace, dtype=float)
    
    return Energy_Trace, Num_particle_Trace, Trail_move_counts_tracker, Lambda
