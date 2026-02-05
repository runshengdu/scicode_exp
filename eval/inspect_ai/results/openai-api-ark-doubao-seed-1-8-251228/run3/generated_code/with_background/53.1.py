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
