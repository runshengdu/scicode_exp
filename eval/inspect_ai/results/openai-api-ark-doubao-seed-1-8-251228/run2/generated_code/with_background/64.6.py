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
    r1_np = np.asarray(r1)
    r2_np = np.asarray(r2)
    delta = r1_np - r2_np
    # Apply minimum image correction to each component
    delta_corrected = delta - L * np.round(delta / L)
    # Calculate Euclidean distance
    distance = np.linalg.norm(delta_corrected)
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
    sigma_over_r = sigma / r
    lj_term = sigma_over_r ** 12 - sigma_over_r ** 6
    return 4 * epsilon * lj_term


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
    for r_j in positions:
        d = dist(r, r_j, L)
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
    # Iterate over all unique unordered pairs of particles
    for r1, r2 in itertools.combinations(positions, 2):
        # Calculate minimum image distance between the pair
        d = dist(r1, r2, L)
        # Add pairwise Lennard-Jones energy to total
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
    # Convert initial positions to a list for dynamic modifications
    positions = np.asarray(initial_positions).tolist()
    # Calculate initial system energy
    initial_energy = E_system(initial_positions, L, sigma, epsilon)
    
    # Initialize traces and tracker
    Energy_Trace = [initial_energy]
    Num_particle_Trace = [len(positions)]
    Trail_move_counts_tracker = {
        'Insertion': [0, 0],
        'Deletion': [0, 0],
        'Move': [0, 0]
    }
    
    # Physical constants
    h = 6.62607015e-34  # Planck's constant in J·s
    k_B = 1.380649e-23   # Boltzmann's constant in J/K
    
    # Precompute beta and thermal de Broglie wavelength
    beta = 1.0 / (k_B * T)
    Lambda = np.sqrt((h**2 * beta) / (2 * np.pi * mass))
    
    for _ in range(num_steps):
        rand = np.random.rand()
        
        if rand < prob_insertion:
            # Attempt particle insertion
            Trail_move_counts_tracker['Insertion'][0] += 1
            # Generate random position within the box, wrapped
            new_pos = np.random.rand(3) * L
            new_pos = wrap(new_pos, L)
            current_energy = Energy_Trace[-1]
            N = len(positions)
            
            # Calculate energy change from insertion
            delta_E = E_i(new_pos, positions, L, sigma, epsilon)
            U_N_plus_1 = current_energy + delta_E
            
            # Compute acceptance probability
            V = L ** 3
            acceptance_term = (V / (Lambda**3 * (N + 1))) * np.exp(-beta * (U_N_plus_1 - current_energy) + beta * mu)
            chi = min(acceptance_term, 1.0)
            
            # Accept or reject the insertion
            if np.random.rand() < chi:
                positions.append(new_pos.tolist())
                Energy_Trace.append(U_N_plus_1)
                Num_particle_Trace.append(N + 1)
                Trail_move_counts_tracker['Insertion'][1] += 1
            else:
                Energy_Trace.append(current_energy)
                Num_particle_Trace.append(N)
                
        elif rand < prob_insertion + prob_deletion:
            # Attempt particle deletion
            Trail_move_counts_tracker['Deletion'][0] += 1
            N = len(positions)
            
            if N == 0:
                # No particles to delete, record unchanged state
                Energy_Trace.append(Energy_Trace[-1])
                Num_particle_Trace.append(0)
                continue
            
            # Select random particle to delete
            idx = np.random.randint(N)
            r_j = positions[idx]
            other_positions = positions[:idx] + positions[idx+1:]
            
            # Calculate energy change from deletion
            delta_E = E_i(r_j, other_positions, L, sigma, epsilon)
            current_energy = Energy_Trace[-1]
            U_N_minus_1 = current_energy - delta_E
            
            # Compute acceptance probability
            V = L ** 3
            acceptance_term = (Lambda**3 * N / V) * np.exp(-beta * (U_N_minus_1 - current_energy) - beta * mu)
            chi = min(acceptance_term, 1.0)
            
            # Accept or reject the deletion
            if np.random.rand() < chi:
                del positions[idx]
                Energy_Trace.append(U_N_minus_1)
                Num_particle_Trace.append(N - 1)
                Trail_move_counts_tracker['Deletion'][1] += 1
            else:
                Energy_Trace.append(current_energy)
                Num_particle_Trace.append(N)
                
        else:
            # Attempt particle displacement
            Trail_move_counts_tracker['Move'][0] += 1
            N = len(positions)
            
            if N == 0:
                # No particles to move, record unchanged state
                Energy_Trace.append(Energy_Trace[-1])
                Num_particle_Trace.append(0)
                continue
            
            # Select random particle to displace
            idx = np.random.randint(N)
            r_old = positions[idx]
            
            # Generate new position with displacement and wrap to box
            displacement = disp_size * L * (np.random.rand(3) - 0.5)
            r_new_unwrapped = np.asarray(r_old) + displacement
            r_new = wrap(r_new_unwrapped, L)
            
            # Calculate energy change from displacement
            other_positions = positions[:idx] + positions[idx+1:]
            energy_old = E_i(r_old, other_positions, L, sigma, epsilon)
            energy_new = E_i(r_new, other_positions, L, sigma, epsilon)
            delta_E = energy_new - energy_old
            current_energy = Energy_Trace[-1]
            
            # Metropolis acceptance criterion
            if delta_E <= 0:
                accept = True
            else:
                accept_prob = np.exp(-beta * delta_E)
                accept = np.random.rand() < accept_prob
            
            # Accept or reject the displacement
            if accept:
                positions[idx] = r_new.tolist()
                new_energy = current_energy + delta_E
                Energy_Trace.append(new_energy)
                Num_particle_Trace.append(N)
                Trail_move_counts_tracker['Move'][1] += 1
            else:
                Energy_Trace.append(current_energy)
                Num_particle_Trace.append(N)
    
    # Convert traces to numpy arrays
    Energy_Trace = np.array(Energy_Trace)
    Num_particle_Trace = np.array(Num_particle_Trace)
    
    return Energy_Trace, Num_particle_Trace, Trail_move_counts_tracker, Lambda
