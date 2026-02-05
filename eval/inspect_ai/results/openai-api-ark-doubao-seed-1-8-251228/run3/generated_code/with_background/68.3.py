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
