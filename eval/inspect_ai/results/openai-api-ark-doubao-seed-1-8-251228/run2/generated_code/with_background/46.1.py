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
        r = np.linalg.norm(configs, axis=-1)  # Shape (nconf, nelec)
        # Sum distances across electrons for each configuration
        sum_r = np.sum(r, axis=1)  # Shape (nconf,)
        # Compute unnormalized wave function
        return np.exp(-self.alpha * sum_r)

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        # Compute distances with keepdims to maintain broadcasting compatibility
        r = np.linalg.norm(configs, axis=-1, keepdims=True)  # Shape (nconf, nelec, 1)
        # Compute gradient normalized by psi
        return -self.alpha * (configs / r)

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        r = np.linalg.norm(configs, axis=-1)  # Shape (nconf, nelec)
        # Compute per-electron laplacian contribution
        return self.alpha ** 2 - (2 * self.alpha) / r

    def kinetic(self, configs):
        '''Calculate the kinetic energy
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        # Get per-electron laplacian contributions
        lap = self.laplacian(configs)
        # Sum contributions across electrons for each configuration
        total_lap = np.sum(lap, axis=1)
        # Local kinetic energy is -0.5 times total laplacian over psi
        return -0.5 * total_lap
