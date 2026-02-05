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
