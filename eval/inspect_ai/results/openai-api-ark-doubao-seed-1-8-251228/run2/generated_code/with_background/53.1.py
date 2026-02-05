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
    # Calculate propensities for each reaction
    a1 = alpha * prey  # Prey birth propensity
    a2 = beta * prey * predator  # Predation propensity
    a3 = gamma * predator  # Predator death propensity
    
    total_a = a1 + a2 + a3
    
    # Check if no reactions can occur
    if total_a <= 0.0:
        return (None, prey, predator, None)
    
    # Sample time step from exponential distribution with rate equal to total propensity
    time_step = np.random.exponential(scale=1.0 / total_a)
    
    # Calculate probabilities for each reaction
    p1 = a1 / total_a
    p2 = a2 / total_a
    
    # Randomly select which reaction to occur
    r = np.random.rand()
    if r < p1:
        # Prey birth event
        prey += 1
        event = "prey_birth"
    elif r < p1 + p2:
        # Predation event: prey dies, predator increases
        prey -= 1
        predator += 1
        event = "predation"
    else:
        # Predator death event
        predator -= 1
        event = "predator_death"
    
    return (time_step, prey, predator, event)
