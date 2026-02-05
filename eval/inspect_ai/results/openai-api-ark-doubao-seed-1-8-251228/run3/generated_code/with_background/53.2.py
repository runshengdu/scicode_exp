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
