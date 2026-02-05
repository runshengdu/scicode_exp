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
        ri = np.linalg.norm(configs, axis=-1)
        sum_ri = np.sum(ri, axis=1)
        return np.exp(-self.alpha * sum_ri)

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        ri = np.linalg.norm(configs, axis=-1)
        ri_expanded = ri[..., np.newaxis]
        unit_vectors = configs / ri_expanded
        return -self.alpha * unit_vectors

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        ri = np.linalg.norm(configs, axis=-1)
        return (-2 * self.alpha / ri) + (self.alpha ** 2)

    def kinetic(self, configs):
        '''Calculate the kinetic energy
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        lap = self.laplacian(configs)
        total_laplacian = np.sum(lap, axis=1)
        return -0.5 * total_laplacian
