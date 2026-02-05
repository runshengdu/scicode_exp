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


def metropolis(configs, wf, tau, nsteps):
    '''Runs metropolis sampling
    Args:
        configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        wf (wavefunction object): Slater class defined before
    Returns:
        poscur (np.array): final electron coordinates after metropolis. Shape (nconf, nelec, ndim)
    '''
    poscur = configs.copy()
    for _ in range(nsteps):
        # Generate proposed positions
        proposal = poscur + np.sqrt(tau) * np.random.normal(size=poscur.shape)
        # Compute wavefunction values for current and proposed positions
        psi_current = wf.value(poscur)
        psi_proposal = wf.value(proposal)
        # Calculate acceptance probability
        accept_prob = (psi_proposal / psi_current) ** 2
        # Determine which configurations to accept
        accept = np.random.rand(*accept_prob.shape) < accept_prob
        # Update positions: accept proposal where condition is true
        poscur = np.where(accept[:, np.newaxis, np.newaxis], proposal, poscur)
    return poscur



def calc_energy(configs, nsteps, tau, alpha, Z):
    '''Args:
        configs (np.array): electron coordinates of shape (nconf, nelec, ndim) where nconf is the number of configurations, nelec is the number of electrons (2 for helium), ndim is the number of spatial dimensions (usually 3)
        nsteps (int): number of Metropolis steps
        tau (float): step size
        alpha (float): exponential decay factor for the wave function
        Z (int): atomic number
    Returns:
        energy (list of float): kinetic energy, electron-ion potential, and electron-electron potential
        error (list of float): error bars of kinetic energy, electron-ion potential, and electron-electron potential
    '''
    # Initialize wave function and Hamiltonian objects
    slater = Slater(alpha)
    ham = Hamiltonian(Z)
    
    # Run Metropolis to obtain equilibrated configurations
    equil_configs = metropolis(configs, slater, tau, nsteps)
    
    # Calculate energy components for each equilibrated configuration
    kinetic_vals = slater.kinetic(equil_configs)
    ei_vals = ham.potential_electron_ion(equil_configs)
    ee_vals = ham.potential_electron_electron(equil_configs)
    
    # Compute mean values of energy components
    mean_kin = np.mean(kinetic_vals)
    mean_ei = np.mean(ei_vals)
    mean_ee = np.mean(ee_vals)
    energy = [mean_kin, mean_ei, mean_ee]
    
    # Compute standard error of the mean for error bars
    nconf = equil_configs.shape[0]
    if nconf > 1:
        err_kin = np.std(kinetic_vals, ddof=1) / np.sqrt(nconf)
        err_ei = np.std(ei_vals, ddof=1) / np.sqrt(nconf)
        err_ee = np.std(ee_vals, ddof=1) / np.sqrt(nconf)
    else:
        # Single configuration has undefined error bar
        err_kin = 0.0
        err_ei = 0.0
        err_ee = 0.0
    error = [err_kin, err_ei, err_ee]
    
    return energy, error
