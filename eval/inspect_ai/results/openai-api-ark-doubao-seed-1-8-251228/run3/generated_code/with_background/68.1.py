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
