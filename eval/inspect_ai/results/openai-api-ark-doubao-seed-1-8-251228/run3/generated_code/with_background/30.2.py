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



class Jastrow:
    def __init__(self, beta=1):
        '''Args: 
            beta: exponential factor for electron-electron correlation
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
        Returns:
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
        r_vec = self.get_r_vec(configs)
        r_ee = self.get_r_ee(configs)
        # Reshape to enable broadcasting with electron coordinates
        factor = self.beta / r_ee[:, np.newaxis]
        grad_e1 = factor * r_vec
        grad_e2 = -grad_e1
        # Stack gradients for each electron along the electron axis
        return np.stack([grad_e1, grad_e2], axis=1)

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array):  (nconf, nelec)        
        '''
        r_ee = self.get_r_ee(configs)
        term = (self.beta / r_ee) * (self.beta * r_ee + 2)
        # Both electrons have identical laplacian values
        return np.stack([term, term], axis=1)
