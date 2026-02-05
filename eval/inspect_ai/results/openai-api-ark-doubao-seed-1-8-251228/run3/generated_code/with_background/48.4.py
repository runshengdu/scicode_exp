import numpy as np
import scipy.interpolate as interpolate

def q_cal(th, gamma, E0, omega):
    '''Calculate the in-plane momentum q, and out-of-plane momenta of the incident and scattered electron k_i_z and k_s_z.
    Ensure that the signs of q, k_i_z and k_s_z are correctly represented.
    Input 
    th, angle between the incident electron and the sample surface normal is 90-th, a list of float in the unit of degree
    gamma, angle between the incident and scattered electron, a list of float in the unit of degree
    E0, incident electron energy, float in the unit of eV
    omega, energy loss, a list of float in the unit of eV
    Output
    Q: a tuple (q,k_i_z,k_s_z) in the unit of inverse angstrom, where q is in-plane momentum, 
       and k_i_z and k_s_z are out-of-plane momenta of the incident and scattered electron    
    '''
    # Physical constants
    m_e_c2 = 510998.95  # eV, m_e*c²
    hc = 12398.4193     # eV·Å, h*c converted from eV·nm
    hbar_c = hc / (2 * np.pi)  # eV·Å, ℏ*c
    
    # Convert input lists to numpy arrays for element-wise operations
    th_arr = np.array(th)
    gamma_arr = np.array(gamma)
    omega_arr = np.array(omega)
    
    # Convert angles from degrees to radians
    th_rad = np.radians(th_arr)
    gamma_rad = np.radians(gamma_arr)
    
    # Calculate incident electron momentum magnitude (Å⁻¹)
    k0 = np.sqrt(2 * m_e_c2 * E0) / hbar_c
    
    # Calculate scattered electron momentum magnitude for each energy loss (Å⁻¹)
    ks_arr = np.sqrt(2 * m_e_c2 * (E0 - omega_arr)) / hbar_c
    
    # Out-of-plane momentum of incident electron (+z direction)
    k_i_z_arr = k0 * np.sin(th_rad)
    
    # In-plane momentum of incident electron (+x direction)
    k_i_x_arr = k0 * np.cos(th_rad)
    
    # In-plane momentum of scattered electron (+x direction)
    k_s_x_arr = ks_arr * np.cos(th_rad - gamma_rad)
    
    # In-plane momentum transfer q = k_s_x - k_i_x
    q_arr = k_s_x_arr - k_i_x_arr
    
    # Out-of-plane momentum of scattered electron (-z direction)
    k_s_z_arr = ks_arr * np.sin(th_rad - gamma_rad)
    
    # Convert numpy arrays back to lists to match input type
    q = q_arr.tolist()
    k_i_z = k_i_z_arr.tolist()
    k_s_z = k_s_z_arr.tolist()
    
    return (q, k_i_z, k_s_z)


def MatELe(th, gamma, E0, omega):
    '''Calculate the Coulomb matrix element in the cgs system using diffractometer angles and the electron energy. 
    For simplicity, assume 4\pi*e^2 = 1 where e is the elementary charge. 
    Input 
    th, angle between the incident electron and the sample surface normal is 90-th, a list of float in the unit of degree
    gamma, angle between the incident and scattered electron, a list of float in the unit of degree
    E0, incident electron energy, float in the unit of eV
    omega, energy loss, a list of float in the unit of eV
    Output
    V_eff: matrix element in the unit of the inverse of square angstrom, a list of float
    '''
    # Retrieve momentum components from q_cal
    q, k_i_z, k_s_z = q_cal(th, gamma, E0, omega)
    
    # Convert lists to numpy arrays for element-wise operations
    q_arr = np.array(q)
    kiz_arr = np.array(k_i_z)
    ksz_arr = np.array(k_s_z)
    
    # Calculate kappa = k_i_z + k_s_z
    kappa_arr = kiz_arr + ksz_arr
    
    # Compute denominator: q² + kappa²
    denominator = q_arr ** 2 + kappa_arr ** 2
    
    # Calculate effective Coulomb matrix element
    v_eff_arr = 1 / denominator
    
    # Convert back to list format
    V_eff = v_eff_arr.tolist()
    
    return V_eff


def S_cal(omega, I, th, gamma, E0):
    '''Convert the experimental data to density-density correlation function, where \sigma_0 = 1 
    Input 
    omega, energy loss, a list of float in the unit of eV
    I, measured cross section from the detector in the unit of Hz, a list of float
    th, angle between the incident electron and the sample surface normal is 90-th, a list of float in the unit of degree
    gamma, angle between the incident and scattered electron, a list of float in the unit of degree
    E0, incident electron energy, float in the unit of eV
    Output
    S_omega: density-density correlation function, a list of float
    '''
    # Calculate effective Coulomb matrix element using provided function
    V_eff = MatELe(th, gamma, E0, omega)
    
    # Convert input lists to numpy arrays for element-wise operations
    V_eff_arr = np.array(V_eff)
    I_arr = np.array(I)
    
    # Compute density-density correlation function S(omega) = I / (V_eff^2) with sigma0 = 1
    S_omega_arr = I_arr / (V_eff_arr ** 2)
    
    # Convert back to list format for output
    S_omega = S_omega_arr.tolist()
    
    return S_omega



def chi_cal(omega, I, th, gamma, E0):
    '''Convert the density-density correlation function to the imaginary part of the density response function 
    by antisymmetrizing S(\omega). Temperature is not required for this conversion.
    Input 
    omega, energy loss, a list of float in the unit of eV
    I, measured cross section from the detector in the unit of Hz, a list of float
    th, angle between the incident electron and the sample surface normal is 90-th, a list of float in the unit of degree
    gamma, angle between the incident and scattered electron, a list of float in the unit of degree
    E0, incident electron energy, float in the unit of eV
    Output
    chi: negative of the imaginary part of the density response function, a list of float
    '''
    # Compute density-density correlation function for each input data point
    S_omega = S_cal(omega, I, th, gamma, E0)
    
    # Convert input lists to numpy arrays for numerical operations
    omega_arr = np.array(omega)
    S_arr = np.array(S_omega)
    
    # Sort omega values and corresponding S values to create a monotonic x array for interpolation
    sorted_indices = np.argsort(omega_arr)
    sorted_omega = omega_arr[sorted_indices]
    sorted_S = S_arr[sorted_indices]
    
    # Create interpolation function for S(omega), returning 0 for energies outside input range
    s_interp = interpolate.interp1d(
        sorted_omega, sorted_S, kind='linear', bounds_error=False, fill_value=0.0
    )
    
    # Calculate S(-omega) for each input omega value
    S_neg_omega = s_interp(-omega_arr)
    
    # Compute the negative imaginary part of the density response function
    chi_arr = np.pi * (S_arr - S_neg_omega)
    
    # Convert back to list format for output
    chi = chi_arr.tolist()
    
    return chi
