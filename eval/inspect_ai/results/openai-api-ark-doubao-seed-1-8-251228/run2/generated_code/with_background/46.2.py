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
        r = np.linalg.norm(configs, axis=-1)  # Shape (nconf, nelec)
        # Sum of inverse distances for each configuration
        sum_inv_r = np.sum(1.0 / r, axis=1)  # Shape (nconf,)
        # Electron-ion potential is -Z * sum(1/r_i)
        return -self.Z * sum_inv_r

    def potential_electron_electron(self, configs):
        '''Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ee (np.array): (nconf,)
        '''
        # Extract coordinates of the two electrons
        e1 = configs[:, 0, :]
        e2 = configs[:, 1, :]
        # Compute distance between the two electrons
        r12 = np.linalg.norm(e1 - e2, axis=-1)  # Shape (nconf,)
        # Electron-electron potential is 1/r12
        return 1.0 / r12

    def potential(self, configs):
        '''Total potential energy
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v (np.array): (nconf,)        
        '''
        return self.potential_electron_ion(configs) + self.potential_electron_electron(configs)
