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
    coord = r_np % L
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
        """Compute the Lennard-Jones potential between a pair of atoms given their distance."""
        if 0 < r_ij < r_c:
            sigma_over_r = sigma / r_ij
            return 4 * epsilon * (sigma_over_r ** 12 - sigma_over_r ** 6)
        else:
            return 0.0
    
    r_np = np.asarray(r)
    pos_np = np.asarray(pos)
    total_energy = 0.0
    
    for pos_j in pos_np:
        # Calculate distance vector between target particle and current particle
        delta = pos_j - r_np
        # Apply minimum image convention to get shortest periodic distance
        delta_mic = delta - L * np.round(delta / L)
        # Compute Euclidean distance
        r_ij = np.linalg.norm(delta_mic)
        # Add pairwise potential energy to total
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
    # Generate random 3D position for the test particle within the simulation box
    r_test = np.random.uniform(0.0, L, size=3)
    
    # Calculate the insertion energy using the pre-defined E_i function
    insertion_energy = E_i(r_test, pos, sigma, epsilon, L, r_c)
    
    # Boltzmann constant (J/K)
    k_B = 1.380649e-23
    beta = 1.0 / (k_B * T)
    
    # Compute the Boltzmann factor
    Boltz = np.exp(-beta * insertion_energy)
    
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
    # Calculate the side length of the cubic box
    L = (N / rho) ** (1/3)
    
    # Determine the smallest integer n such that n^3 >= N to form a cubic grid
    n = int(N ** (1/3))
    while n ** 3 < N:
        n += 1
    
    # Calculate the spacing between adjacent particles along each axis
    dx = L / n
    
    # Generate coordinates along each axis, centered in each grid cell
    x = np.arange(n) * dx + dx / 2
    y = np.arange(n) * dx + dx / 2
    z = np.arange(n) * dx + dx / 2
    
    # Create a 3D meshgrid of positions
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Flatten the meshgrid to get a 2D array of positions
    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    
    # Select exactly N particles from the grid
    positions = positions[:N]
    
    return positions, L
