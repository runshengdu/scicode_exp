import numpy as np
from scipy.interpolate import interp1d
from numpy.fft import fft, fftfreq

def gillespie_step(prey, predator, alpha, beta, gamma):
    '''Perform one step of the Gillespie simulation for a predator-prey system.
    Input:
    prey: current population of prey, integer
    predator: current population of predators, integer
    alpha: prey birth rate, float
    beta: predation rate, float
    gamma: predator death rate, float
    Output:
    time_step: time duration until next event occurs, a float; None if no event occurs
    prey: updated population of prey, integer
    predator: updated population of predators, integer
    event: a string describing the event that occurrs ("prey_birth", "predation", or "predator_death"); None if no event occurs
    '''
    # Calculate reaction propensities
    a1 = alpha * prey          # Propensity for prey birth
    a2 = beta * prey * predator # Propensity for predation
    a3 = gamma * predator      # Propensity for predator death
    
    total_propensity = a1 + a2 + a3
    
    # Handle case where no reactions can occur
    if total_propensity < 1e-12:
        return (None, prey, predator, None)
    
    # Sample time until next reaction from exponential distribution
    time_step = np.random.exponential(scale=1.0 / total_propensity)
    
    # Randomly select which reaction occurs
    random_sample = np.random.uniform(0, total_propensity)
    if random_sample < a1:
        # Prey birth event
        prey += 1
        event = "prey_birth"
    elif random_sample < a1 + a2:
        # Predation event: prey decreases, predator increases
        prey -= 1
        predator += 1
        event = "predation"
    else:
        # Predator death event
        predator -= 1
        event = "predator_death"
    
    return (time_step, prey, predator, event)


def evolve_LV(prey, predator, alpha, beta, gamma, T):
    '''Simulate the predator-prey dynamics using the Gillespie simulation algorithm.
    This function tracks and records the populations of prey and predators and the times at which changes occur.
    Input:
    prey: initial population of prey, integer
    predator: initial population of predators, integer
    alpha: prey birth rate, float
    beta: predation rate, float
    gamma: predator death rate, float
    T: total time of the simulation, float
    Output:
    time_cor: time coordinates of population evolution, 1D array of floats
    prey_evol: evolution history of prey population, 1D array of floats (same size as time_cor)
    predator_evol: evolution history of predator population, 1D array of floats (same size as time_cor)
    eco_event: A string describing the ecological event ("coexistence", "predator extinction", or "mutual extinction").
    '''
    # Initialize tracking variables
    current_time = 0.0
    current_prey = prey
    current_predator = predator
    
    time_cor = [current_time]
    prey_evol = [current_prey]
    predator_evol = [current_predator]
    
    while True:
        # Perform one Gillespie step
        time_step, new_prey, new_predator, event = gillespie_step(current_prey, current_predator, alpha, beta, gamma)
        
        # Check if no event can occur
        if time_step is None:
            break
        
        # Calculate next event time
        next_time = current_time + time_step
        
        # Check if next event is beyond the final time T
        if next_time > T:
            break
        
        # Update current state and trackers
        current_time = next_time
        current_prey = new_prey
        current_predator = new_predator
        
        time_cor.append(current_time)
        prey_evol.append(current_prey)
        predator_evol.append(current_predator)
    
    # Convert lists to numpy arrays
    time_cor = np.array(time_cor, dtype=float)
    prey_evol = np.array(prey_evol, dtype=float)
    predator_evol = np.array(predator_evol, dtype=float)
    
    # Determine the ecological event based on final state
    final_prey = prey_evol[-1]
    final_predator = predator_evol[-1]
    
    if final_prey == 0:
        eco_event = "mutual extinction"
    elif final_predator == 0:
        eco_event = "predator extinction"
    else:
        eco_event = "coexistence"
    
    return time_cor, prey_evol, predator_evol, eco_event


def spectral_periodicity(t, population):
    '''Estimate the periodicity of population with uneven time step and stochasticity.
    Input:
    t: time coordinates of population evolution, 1D array of floats
    population: evolution history of population of some species, 1D array of floats (same size as t)
    Output:
    periodicity: estimated periodicity, float rounded up to one decimal point.
    '''
    # Handle edge cases where periodicity can't be estimated
    if len(t) != len(population) or len(t) <= 1:
        return 0.0
    
    t_start = t[0]
    t_end = t[-1]
    total_duration = t_end - t_start
    
    if total_duration < 1e-12:
        return 0.0
    
    # Compute minimum time between consecutive events to determine sampling step
    dt_events = np.diff(t)
    dt_min = np.min(dt_events)
    
    # Create regular time grid with step size equal to minimum event interval
    t_reg = np.arange(t_start, t_end + dt_min, dt_min)
    
    # Sample the step function population onto the regular grid
    indices = np.digitize(t_reg, t) - 1
    indices = np.clip(indices, 0, len(population) - 1)
    population_reg = population[indices]
    
    # Remove linear trend from the data
    A = np.vstack([t_reg, np.ones(len(t_reg))]).T
    slope, intercept = np.linalg.lstsq(A, population_reg, rcond=None)[0]
    trend = slope * t_reg + intercept
    population_detrended = population_reg - trend
    
    # Compute FFT
    n_samples = len(population_detrended)
    dt_reg = dt_min  # Sampling interval of regular grid
    yf = fft(population_detrended)
    xf = fftfreq(n_samples, d=dt_reg)
    
    # Focus on positive frequencies
    positive_mask = xf > 0
    xf_pos = xf[positive_mask]
    yf_pos = np.abs(yf[positive_mask])
    
    # Check if there's any oscillatory component
    if np.max(yf_pos) < 1e-12:
        return 0.0
    
    # Smooth the frequency spectrum to reduce noise
    window_size = 5
    if len(yf_pos) >= window_size:
        window = np.ones(window_size) / window_size
        yf_smoothed = np.convolve(yf_pos, window, mode='same')
        dominant_idx = np.argmax(yf_smoothed)
    else:
        dominant_idx = np.argmax(yf_pos)
    
    # Refine dominant frequency using quadratic interpolation around the peak
    if 0 < dominant_idx < len(xf_pos) - 1:
        k = dominant_idx
        x_prev, x_curr, x_next = xf_pos[k-1], xf_pos[k], xf_pos[k+1]
        y_prev, y_curr, y_next = yf_pos[k-1], yf_pos[k], yf_pos[k+1]
        
        dx = x_curr - x_prev
        denominator = 2 * (y_prev - 2 * y_curr + y_next)
        
        if abs(denominator) < 1e-12:
            refined_freq = x_curr
        else:
            t = (y_prev - y_next) / denominator
            refined_freq = x_curr + t * dx
        # Ensure refined frequency stays within valid positive range
        refined_freq = max(refined_freq, xf_pos[0])
    else:
        refined_freq = xf_pos[dominant_idx]
    
    periodicity = 1.0 / refined_freq
    
    # Round to one decimal point
    periodicity_rounded = round(periodicity, 1)
    
    return periodicity_rounded



def predator_prey(prey, predator, alpha, beta, gamma, T):
    '''Simulate the predator-prey dynamics using the Gillespie simulation algorithm.
    Records the populations of prey and predators and the times at which changes occur.
    Analyze the ecological phenomenon happens in the system.
    Input:
    prey: initial population of prey, integer
    predator: initial population of predators, integer
    alpha: prey birth rate, float
    beta: predation rate, float
    gamma: predator death rate, float
    T: total time of the simulation, float
    Output:
    time_cor: time coordinates of population evolution, 1D array of floats
    prey_evol: evolution history of prey population, 1D array of floats (same size as time_cor)
    predator_evol: evolution history of predator population, 1D array of floats (same size as time_cor)
    eco_event: A string describing the ecological event ("coexistence", "predator extinction", or "mutual extinction").
    prey_period: estimated periodicity of prey population, float rounded up to one decimal point; 0.0 if no coexistence
    predator_period: estimated periodicity of redator population, float rounded up to one decimal point; 0.0 if no coexistence
    '''
    # Run the Gillespie simulation to get population dynamics and ecological event
    time_cor, prey_evol, predator_evol, eco_event = evolve_LV(prey, predator, alpha, beta, gamma, T)
    
    # Initialize periodicity values to 0.0 (default for non-coexistence cases)
    prey_period = 0.0
    predator_period = 0.0
    
    # Calculate periodicity only if coexistence occurs
    if eco_event == "coexistence":
        prey_period = spectral_periodicity(time_cor, prey_evol)
        predator_period = spectral_periodicity(time_cor, predator_evol)
    
    return time_cor, prey_evol, predator_evol, eco_event, prey_period, predator_period
