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
