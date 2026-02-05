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
    # Fundamental constants with correct unit conversions
    m_e_c2 = 510998.95  # eV, m_e*c² = 0.51099895 MeV converted to eV
    hc = 1239.84193     # eV·nm, given Planck's constant times speed of light
    hbar_c_ang = (hc / (2 * np.pi)) * 10  # Convert hbar*c from eV·nm to eV·Å (1 nm = 10 Å)
    
    # Precompute constant factor for wavevector calculation (Å⁻¹ / sqrt(eV))
    sqrt_2mec2 = np.sqrt(2 * m_e_c2)
    k_const = sqrt_2mec2 / hbar_c_ang
    
    # Convert input lists to numpy arrays for vectorized operations
    th_arr = np.array(th)
    gamma_arr = np.array(gamma)
    omega_arr = np.array(omega)
    
    # Calculate incident angle relative to surface normal (theta_i) in radians
    theta_i_deg = 90 - th_arr
    theta_i_rad = np.deg2rad(theta_i_deg)
    
    # Calculate scattered angle relative to surface normal (theta_s) in radians
    theta_s_deg = theta_i_deg + gamma_arr
    theta_s_rad = np.deg2rad(theta_s_deg)
    
    # Calculate magnitudes of incident and scattered electron wavevectors
    k_i = k_const * np.sqrt(E0)
    k_s = k_const * np.sqrt(E0 - omega_arr)
    
    # Compute out-of-plane (z) components
    k_i_z = k_i * np.cos(theta_i_rad)  # Positive since incident z-component is along +z
    k_s_z = k_s * np.cos(theta_s_rad)  # Negative since scattered z-component is along -z
    
    # Compute in-plane (x) components and momentum transfer
    k_i_x = k_i * np.sin(theta_i_rad)
    k_s_x = k_s * np.sin(theta_s_rad)
    q = k_s_x - k_i_x  # In-plane momentum transfer
    
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
    # Retrieve in-plane and out-of-plane momenta from q_cal
    q, k_i_z, k_s_z = q_cal(th, gamma, E0, omega)
    
    # Compute kappa as the sum of out-of-plane momenta
    kappa = k_i_z + k_s_z
    
    # Calculate the denominator term
    denominator = q ** 2 + kappa ** 2
    
    # Compute effective Coulomb matrix element with 4πe² = 1
    v_eff_array = 1 / denominator
    
    # Convert to list of floats as required
    V_eff = v_eff_array.tolist()
    
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
    # Calculate effective Coulomb matrix element
    V_eff = MatELe(th, gamma, E0, omega)
    
    # Convert input lists to numpy arrays for vectorized operations
    I_arr = np.array(I)
    V_eff_arr = np.array(V_eff)
    
    # Compute density-density correlation function using the given relation
    S_omega_arr = I_arr / (V_eff_arr ** 2)
    
    # Convert back to list format as required
    S_omega = S_omega_arr.tolist()
    
    return S_omega
