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



class Jastrow:
    def __init__(self, beta=1):
        '''Args: 
            beta: electron-electron interaction strength
        '''
        self.beta = beta

    def get_r_vec(self, configs):
        '''Returns a vector pointing from r2 to r1, which is r_12 = [x1 - x2, y1 - y2, z1 - z2].
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            r_vec (np.array): (nconf, ndim)
        '''
        return configs[:, 0, :] - configs[:, 1, :]

    def get_r_ee(self, configs):
        '''Returns the Euclidean distance from r2 to r1
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            r_ee (np.array): (nconf,)
        '''
        r_vec = self.get_r_vec(configs)
        return np.linalg.norm(r_vec, axis=-1)

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
        r_vec = self.get_r_vec(configs)
        r_ee = self.get_r_ee(configs)
        # Compute scaling factor and broadcast to match r_vec dimensions
        factor = self.beta / r_ee[:, np.newaxis]
        # Gradient contributions for each electron
        grad_e0 = factor * r_vec
        grad_e1 = -factor * r_vec
        # Stack to get (nconf, nelec, ndim) shape
        return np.stack([grad_e0, grad_e1], axis=1)

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array):  (nconf, nelec)        
        '''
        r_ee = self.get_r_ee(configs)
        # Compute the laplacian term for each configuration
        term = (self.beta / r_ee) * (self.beta * r_ee + 2)
        # Expand to (nconf, nelec) by repeating for each electron
        return np.repeat(term[:, np.newaxis], 2, axis=1)


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
        return self.wf1.value(configs) * self.wf2.value(configs)

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi of the multiplication of two wave functions
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        return self.wf1.gradient(configs) + self.wf2.gradient(configs)

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
        dot_product = np.sum(grad1 * grad2, axis=-1)
        return lap1 + lap2 + 2 * dot_product

    def kinetic(self, configs):
        '''Calculate the kinetic energy of the multiplication of two wave functions
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        total_laplacian = np.sum(self.laplacian(configs), axis=1)
        return -0.5 * total_laplacian


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
        # Calculate distance of each electron from the nucleus
        ri = np.linalg.norm(configs, axis=-1)
        # Sum 1/ri over electrons and multiply by -Z
        return -self.Z * np.sum(1.0 / ri, axis=1)

    def potential_electron_electron(self, configs):
        '''Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ee (np.array): (nconf,)
        '''
        # Vector between electron 0 and electron 1
        r12_vec = configs[:, 0, :] - configs[:, 1, :]
        # Distance between electron 0 and electron 1
        r12 = np.linalg.norm(r12_vec, axis=-1)
        # Electron-electron potential is 1/r12
        return 1.0 / r12

    def potential(self, configs):
        '''Total potential energy
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v (np.array): (nconf,)        
        '''
        v_ei = self.potential_electron_ion(configs)
        v_ee = self.potential_electron_electron(configs)
        return v_ei + v_ee


def metropolis(configs, wf, tau=0.01, nsteps=2000):
    '''Runs metropolis sampling
    Args:
        configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        wf (wavefunction object):  MultiplyWF class      
    Returns:
        poscur (np.array): final electron coordinates after metropolis. Shape (nconf, nelec, ndim)
    '''
    poscur = configs.copy()
    for _ in range(nsteps):
        # Propose new positions
        delta = np.random.normal(0.0, np.sqrt(tau), size=poscur.shape)
        pos_proposed = poscur + delta
        
        # Calculate acceptance probability
        psi_old = wf.value(poscur)
        psi_new = wf.value(pos_proposed)
        ratio = (psi_new / psi_old) ** 2
        accept_prob = np.minimum(ratio, 1.0)
        
        # Generate random acceptance criteria
        u = np.random.uniform(0.0, 1.0, size=poscur.shape[0])
        
        # Update positions for accepted moves
        accept = u < accept_prob
        poscur[accept] = pos_proposed[accept]
    
    return poscur


def get_acceptance_ratio(configs_old, configs_new, drift_old, drift_new, dtau, wf):
    '''Args:
        configs_old (np.array): electron positions before move (nconf, nelec, ndim)
        configs_new (np.array): electron positions after  move (nconf, nelec, ndim)
        drift_old (np.array): gradient calculated on old configs multiplied by dtau (nconf, nelec, ndim)
        drift_new (np.array): gradient calculated on new configs (nconf, nelec, ndim)
        dtau (float): time step
        wf (wave function object): MultiplyWF class
    Returns:
        acceptance_ratio (nconf,):
    '''
    # Calculate squared ratio of wave function values
    psi_old = wf.value(configs_old)
    psi_new = wf.value(configs_new)
    psi_ratio_sq = (psi_new / psi_old) ** 2
    
    # Compute coordinate difference between new and old configurations
    delta_R = configs_new - configs_old
    
    # Calculate terms from Green's function Gaussian components
    term_old_new = np.sum((delta_R - drift_old) ** 2, axis=(-2, -1))
    term_new_old = np.sum((delta_R + drift_new * dtau) ** 2, axis=(-2, -1))
    
    # Compute exponential factor from Green's function ratio
    exponent = (term_old_new - term_new_old) / (2 * dtau)
    exp_term = np.exp(exponent)
    
    # Combine factors to get acceptance ratio
    acc_ratio = psi_ratio_sq * exp_term
    
    return acc_ratio




def branch(weight):
    '''Performs DMC branching.
    Args:
        weight (list or np.array): list of weights. Shape (nconfig,)
    Return:
        new_indices (list or np.array): indices of chosen configurations. Shape (nconfig,)
    '''
    w = np.asarray(weight)
    total_weight = w.sum()
    
    if total_weight == 0:
        # If all weights are zero, sample uniformly
        return np.random.choice(len(w), size=len(w), replace=True)
    
    # Normalize weights to probabilities and ensure exact summation to 1
    probabilities = w / total_weight
    probabilities /= probabilities.sum()  # Safeguard against floating point errors
    
    # Sample indices with replacement according to probabilities
    new_indices = np.random.choice(len(w), size=len(w), replace=True, p=probabilities)
    
    return new_indices
