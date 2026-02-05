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
        norms = np.linalg.norm(configs, axis=2)  # Shape (nconf, nelec)
        total = -self.alpha * np.sum(norms, axis=1)  # Shape (nconf,)
        return np.exp(total)

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        norms = np.linalg.norm(configs, axis=2, keepdims=True)  # Shape (nconf, nelec, 1)
        return -self.alpha * (configs / norms)

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        norms = np.linalg.norm(configs, axis=2)  # Shape (nconf, nelec)
        return (-2 * self.alpha) / norms + (self.alpha ** 2)

    def kinetic(self, configs):
        '''Calculate the kinetic energy / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        lap = self.laplacian(configs)
        total_lap = np.sum(lap, axis=1)
        return -0.5 * total_lap
