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
        # Compute distance of each electron from origin
        r = np.linalg.norm(configs, axis=2)
        # Sum distances per configuration and compute exponential
        sum_r = r.sum(axis=1)
        return np.exp(-self.alpha * sum_r)

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        # Compute distance with keepdims to maintain broadcasting
        r = np.linalg.norm(configs, axis=2, keepdims=True)
        # Unit vector scaled by -alpha
        return -self.alpha * (configs / r)

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        # Compute distance of each electron from origin
        r = np.linalg.norm(configs, axis=2)
        alpha = self.alpha
        return (-2 * alpha / r) + (alpha ** 2)

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
        total_lap = lap.sum(axis=1)
        # Kinetic energy is -0.5 times total laplacian ratio
        return -0.5 * total_lap



class Jastrow:
    def __init__(self, beta=1):
        '''Args: 
            beta: Jastrow factor parameter
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
        '''Calculate Jastrow factor
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns 
            jast (np.array): (nconf,)
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
        beta = self.beta
        r_vec = self.get_r_vec(configs)
        r_ee = self.get_r_ee(configs)
        
        # Compute scaling factor with broadcasting support
        factor = beta / r_ee[:, np.newaxis]
        grad_e0 = factor * r_vec
        grad_e1 = -factor * r_vec
        
        # Stack gradients for each electron
        return np.stack([grad_e0, grad_e1], axis=1)

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array):  (nconf, nelec)        
        '''
        beta = self.beta
        r_ee = self.get_r_ee(configs)
        
        # Compute the laplacian term for each configuration
        term = (beta / r_ee) * (beta * r_ee + 2)
        # Both electrons have the same term
        return np.stack([term, term], axis=1)


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
        # Compute dot product of gradients over the spatial dimension axis
        grad_dot = np.sum(grad1 * grad2, axis=2)
        return lap1 + 2 * grad_dot + lap2

    def kinetic(self, configs):
        '''Calculate the kinetic energy of the multiplication of two wave functions
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        lap = self.laplacian(configs)
        total_laplacian = lap.sum(axis=1)
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
        # Compute distance of each electron from the origin
        r = np.linalg.norm(configs, axis=2)
        # Sum inverse distances per configuration and apply potential formula
        return -self.Z * np.sum(1 / r, axis=1)

    def potential_electron_electron(self, configs):
        '''Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ee (np.array): (nconf,)
        '''
        # Vector between electron 1 and electron 2
        r12_vec = configs[:, 0, :] - configs[:, 1, :]
        # Distance between the two electrons
        r12 = np.linalg.norm(r12_vec, axis=1)
        # Electron-electron potential term
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
        # Propose new positions with Gaussian displacement
        displacement = np.sqrt(tau) * np.random.normal(size=poscur.shape)
        pos_new = poscur + displacement
        
        # Calculate wavefunction values for current and proposed positions
        psi_current = wf.value(poscur)
        psi_proposed = wf.value(pos_new)
        
        # Compute acceptance probability ratio
        acceptance_ratio = (psi_proposed / psi_current) ** 2
        
        # Generate uniform random numbers for each configuration
        random_uniform = np.random.rand(poscur.shape[0])
        
        # Accept or reject the proposed positions
        accept_mask = random_uniform < acceptance_ratio
        poscur[accept_mask] = pos_new[accept_mask]
    
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
    # Calculate squared norms for Gaussian terms
    term_old_new = np.sum((configs_new - configs_old - drift_old)**2, axis=(1, 2))
    term_new_old = np.sum((configs_old - configs_new - dtau * drift_new)**2, axis=(1, 2))
    
    # Compute Gaussian component of acceptance ratio
    gaussian_exponent = (term_old_new - term_new_old) / (2 * dtau)
    gaussian_ratio = np.exp(gaussian_exponent)
    
    # Compute squared wavefunction value ratio
    psi_old = wf.value(configs_old)
    psi_new = wf.value(configs_new)
    psi_ratio_sq = (psi_new / psi_old)**2
    
    # Combine components to get total acceptance ratio
    acc_ratio = psi_ratio_sq * gaussian_ratio
    
    return acc_ratio


def branch(weight):
    '''Performs DMC branching.
    Args:
        weight (list or np.array): list of weights. Shape (nconfig,)
    Return:
        new_indices (list or np.array): indices of chosen configurations. Shape (nconfig,)
    '''
    weight = np.asarray(weight)
    nconfig = weight.shape[0]
    
    # Normalize weights to probability distribution
    sum_weights = np.sum(weight)
    norm_weights = weight / sum_weights
    
    # Compute cumulative distribution function
    cdf = np.cumsum(norm_weights)
    # Ensure final CDF value is exactly 1 to avoid out-of-bounds issues
    cdf[-1] = 1.0
    
    # Generate uniform random numbers for sampling
    random_vals = np.random.rand(nconfig)
    
    # Find indices of configurations to keep using searchsorted
    new_indices = np.searchsorted(cdf, random_vals)
    
    return new_indices



def run_dmc(ham, wf, configs, tau, nstep):
    '''Run DMC
    Args:
        ham (hamiltonian object):
        wf (wavefunction object):
        configs (np.array): electron positions before move (nconf, nelec, ndim)
        tau: time step
        nstep: total number of iterations        
    Returns:
        list of local energies
    '''
    poscur = configs.copy()
    # Compute initial local energy
    eloc_prev = wf.kinetic(poscur) + ham.potential(poscur)
    energies = []
    # Initialize trial energy as average of initial local energy
    E_T = np.mean(eloc_prev)
    
    for _ in range(nstep):
        # Propose new positions with drift and diffusion
        drift_old = tau * wf.gradient(poscur)
        displacement = np.sqrt(tau) * np.random.normal(size=poscur.shape)
        pos_new = poscur + drift_old + displacement
        
        # Calculate acceptance ratio
        drift_new = wf.gradient(pos_new)
        acc_ratio = get_acceptance_ratio(poscur, pos_new, drift_old, drift_new, tau, wf)
        
        # Accept or reject moves
        accept = np.random.rand(poscur.shape[0]) < acc_ratio
        poscur[accept] = pos_new[accept]
        
        # Compute new local energies
        eloc_new = wf.kinetic(poscur) + ham.potential(poscur)
        energies.append(eloc_new)
        
        # Compute branching weights
        avg_eloc = (eloc_prev + eloc_new) / 2
        weights = np.exp(-tau * (avg_eloc - E_T))
        
        # Update trial energy
        avg_weights = np.mean(weights)
        E_T -= np.log(avg_weights)
        
        # Perform branching
        new_indices = branch(weights)
        poscur = poscur[new_indices]
        eloc_prev = eloc_new[new_indices]
    
    return energies
