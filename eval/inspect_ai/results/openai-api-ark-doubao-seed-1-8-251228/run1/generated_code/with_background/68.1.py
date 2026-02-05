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
