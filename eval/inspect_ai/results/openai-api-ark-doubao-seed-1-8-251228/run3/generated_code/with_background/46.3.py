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
        r = np.linalg.norm(configs, axis=2)  # shape (nconf, nelec)
        # Sum over electrons for each configuration and compute psi
        sum_r = np.sum(r, axis=1)  # shape (nconf,)
        val = np.exp(-self.alpha * sum_r)
        return val

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        # Compute distance of each electron from origin with preserved dimensions
        r = np.linalg.norm(configs, axis=2, keepdims=True)  # shape (nconf, nelec, 1)
        # Calculate gradient contribution per electron coordinate
        grad = -self.alpha * configs / r
        return grad

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        # Compute distance of each electron from origin
        r = np.linalg.norm(configs, axis=2)  # shape (nconf, nelec)
        # Calculate per-electron laplacian term
        lap = (-2 * self.alpha / r) + (self.alpha ** 2)
        return lap

    def kinetic(self, configs):
        '''Calculate the kinetic energy
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        # Get per-electron laplacian contributions
        lap = self.laplacian(configs)
        # Sum over electrons and apply kinetic energy factor
        kin = -0.5 * np.sum(lap, axis=1)
        return kin


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
        # Compute distance of each electron from origin
        r = np.linalg.norm(configs, axis=2)  # shape (nconf, nelec)
        # Sum electron-ion potential terms for each configuration
        v_ei = -self.Z * np.sum(1.0 / r, axis=1)
        return v_ei

    def potential_electron_electron(self, configs):
        '''Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ee (np.array): (nconf,)
        '''
        # Extract coordinates of the two electrons
        r1 = configs[:, 0, :]
        r2 = configs[:, 1, :]
        # Compute distance between the two electrons
        r12 = np.linalg.norm(r1 - r2, axis=1)  # shape (nconf,)
        # Calculate electron-electron potential
        v_ee = 1.0 / r12
        return v_ee

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



def metropolis(configs, wf, tau, nsteps):
    '''Runs metropolis sampling
    Args:
        configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        wf (wavefunction object): Slater class defined before
    Returns:
        poscur (np.array): final electron coordinates after metropolis. Shape (nconf, nelec, ndim)
    '''
    poscur = configs.copy()
    nconf = poscur.shape[0]
    
    for _ in range(nsteps):
        # Generate proposed positions with Gaussian noise
        noise = np.random.normal(scale=np.sqrt(tau), size=poscur.shape)
        proposal = poscur + noise
        
        # Calculate wavefunction values for current and proposed configurations
        psi_current = wf.value(poscur)
        psi_proposal = wf.value(proposal)
        
        # Compute acceptance probability (ratio of squared magnitudes)
        accept_prob = (psi_proposal / psi_current) ** 2
        
        # Generate uniform random numbers for acceptance check
        u = np.random.rand(nconf)
        
        # Accept or reject each configuration
        accept_mask = u < accept_prob
        poscur[accept_mask] = proposal[accept_mask]
    
    return poscur
