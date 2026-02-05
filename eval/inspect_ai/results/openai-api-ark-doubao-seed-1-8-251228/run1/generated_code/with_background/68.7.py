import numpy as np


class Slater:
    def __init__(self, alpha):
        '''Args: 
            alpha: exponential decay factor
        '''
        self.alpha = alpha

    def value(self, configs):
        '''Calculate unnormalized psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            val (np.array): (nconf,)
        '''
        # Compute distance from origin for each electron in each configuration
        r = np.linalg.norm(configs, axis=2)  # Shape (nconf, nelec)
        # Sum over electrons and compute exponential
        sum_r = np.sum(r, axis=1)  # Shape (nconf,)
        return np.exp(-self.alpha * sum_r)

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        # Compute distance from origin for each electron, keep dimensions for broadcasting
        r = np.linalg.norm(configs, axis=2, keepdims=True)  # Shape (nconf, nelec, 1)
        # Calculate unit vectors and scale by -alpha
        unit_vec = configs / r
        return -self.alpha * unit_vec

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        # Compute distance from origin for each electron
        r = np.linalg.norm(configs, axis=2)  # Shape (nconf, nelec)
        # Calculate per-electron laplacian term
        term = (-2 * self.alpha / r) + (self.alpha ** 2)
        return term

    def kinetic(self, configs):
        '''Calculate the kinetic energy
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        # Get per-electron laplacian contributions
        lap = self.laplacian(configs)
        # Sum over electrons to get total laplacian ratio
        total_lap = np.sum(lap, axis=1)
        # Local kinetic energy is -1/2 * total laplacian ratio
        return -0.5 * total_lap


class Jastrow:
    def __init__(self, beta=1):
        '''Args: 
            beta: exponential factor for electron-electron interaction
        '''
        self.beta = beta

    def get_r_vec(self, configs):
        '''Returns a vector pointing from r2 to r1, which is r_12 = [x1 - x2, y1 - y2, z1 - z2].
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            r_vec (np.array): (nconf, ndim)
        '''
        r1 = configs[:, 0, :]
        r2 = configs[:, 1, :]
        return r1 - r2

    def get_r_ee(self, configs):
        '''Returns the Euclidean distance from r2 to r1
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            r_ee (np.array): (nconf,)
        '''
        r_vec = self.get_r_vec(configs)
        return np.linalg.norm(r_vec, axis=1)

    def value(self, configs):
        '''Calculate unnormalized psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            val (np.array): (nconf,)
        '''
        r_ee = self.get_r_ee(configs)
        return np.exp(self.beta * r_ee)

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        r_vec = self.get_r_vec(configs)
        r_ee = self.get_r_ee(configs)
        factor = self.beta / r_ee[:, np.newaxis]
        grad = np.zeros_like(configs)
        grad[:, 0, :] = factor * r_vec
        grad[:, 1, :] = -factor * r_vec
        return grad

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array):  (nconf, nelec)        
        '''
        r_ee = self.get_r_ee(configs)
        term = (self.beta / r_ee) * (self.beta * r_ee + 2)
        lap = np.stack([term, term], axis=1)
        return lap


class MultiplyWF:
    def __init__(self, wf1, wf2):
        '''Args:
            wf1 (wavefunction object): 
            wf2 (wavefunction object):            
        '''
        self.wf1 = wf1
        self.wf2 = wf2

    def value(self, configs):
        '''Multiply two wave function values
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            val (np.array): (nconf,)
        '''
        val1 = self.wf1.value(configs)
        val2 = self.wf2.value(configs)
        return val1 * val2

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi of the multiplication of two wave functions
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        grad1 = self.wf1.gradient(configs)
        grad2 = self.wf2.gradient(configs)
        return grad1 + grad2

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi of the multiplication of two wave functions
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        lap1 = self.wf1.laplacian(configs)
        lap2 = self.wf2.laplacian(configs)
        grad1 = self.wf1.gradient(configs)
        grad2 = self.wf2.gradient(configs)
        
        # Compute dot product of gradients over dimensions for each electron
        cross_term = 2 * np.sum(grad1 * grad2, axis=2)
        return lap1 + cross_term + lap2

    def kinetic(self, configs):
        '''Calculate the kinetic energy of the multiplication of two wave functions
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        lap = self.laplacian(configs)
        total_laplacian = np.sum(lap, axis=1)
        return -0.5 * total_laplacian


class Hamiltonian:
    def __init__(self, Z):
        '''Z: atomic number
        '''
        self.Z = Z

    def potential_electron_ion(self, configs):
        '''Calculate electron-ion potential
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ei (np.array): (nconf,)
        '''
        r = np.linalg.norm(configs, axis=2)  # Shape (nconf, nelec)
        return -self.Z * np.sum(1 / r, axis=1)

    def potential_electron_electron(self, configs):
        '''Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ee (np.array): (nconf,)
        '''
        r1 = configs[:, 0, :]
        r2 = configs[:, 1, :]
        r_vec = r1 - r2
        r12 = np.linalg.norm(r_vec, axis=1)
        return 1 / r12

    def potential(self, configs):
        '''Total potential energy
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v (np.array): (nconf,)        
        '''
        v_ei = self.potential_electron_ion(configs)
        v_ee = self.potential_electron_electron(configs)
        return v_ei + v_ee


def metropolis(configs, wf, tau=0.01, nsteps=2000):
    '''Runs metropolis sampling
    Args:
        configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        wf (wavefunction object):  MultiplyWF class      
    Returns:
        poscur (np.array): final electron coordinates after metropolis. Shape (nconf, nelec, ndim)
    '''
    poscur = configs.copy()
    for _ in range(nsteps):
        # Generate proposed positions
        proposal = poscur + np.sqrt(tau) * np.random.normal(size=poscur.shape)
        # Compute wavefunction values for current and proposed positions
        psi_old = wf.value(poscur)
        psi_new = wf.value(proposal)
        # Calculate acceptance probability
        acceptance_ratio = (psi_new / psi_old) ** 2
        # Generate uniform random numbers for acceptance check
        u = np.random.uniform(0, 1, size=psi_old.shape)
        # Accept or reject proposals
        accept = u <= acceptance_ratio
        poscur[accept] = proposal[accept]
    return poscur



def get_acceptance_ratio(configs_old, configs_new, drift_old, drift_new, dtau, wf):
    '''Args:
        configs_old (np.array): electron positions before move (nconf, nelec, ndim)
        configs_new (np.array): electron positions after  move (nconf, nelec, ndim)
        drift_old (np.array): gradient calculated on old configs multiplied by dtau (nconf, nelec, ndim)
        drift_new (np.array): gradient calculated on new configs (nconf, nelec, ndim)
        dtau (float): time step
        wf (wave function object): MultiplyWF class
    Returns:
        acceptance_ratio (nconf,):
    '''
    # Compute position difference between new and old configurations
    delta_R = configs_new - configs_old
    
    # Calculate sum of squared terms for forward move (old -> new)
    term_forward = delta_R - drift_old
    sum_forward = np.sum(term_forward ** 2, axis=(1, 2))
    
    # Calculate sum of squared terms for reverse move (new -> old)
    term_reverse = configs_old - configs_new - dtau * drift_new
    sum_reverse = np.sum(term_reverse ** 2, axis=(1, 2))
    
    # Compute the exponent for the acceptance ratio
    exponent = (sum_forward - sum_reverse) / (2 * dtau)
    
    # Calculate final acceptance ratio
    acc_ratio = np.exp(exponent)
    
    return acc_ratio



def branch(weight):
    '''Performs DMC branching.
    Args:
        weight (list or np.array): list of weights. Shape (nconfig,)
    Return:
        new_indices (list or np.array): indices of chosen configurations. Shape (nconfig,)
    '''
    weight = np.asarray(weight)
    nconf = weight.shape[0]
    # Normalize weights to probabilities
    prob = weight / weight.sum()
    # Sample indices with replacement according to probabilities
    new_indices = np.random.choice(nconf, size=nconf, p=prob)
    return new_indices
