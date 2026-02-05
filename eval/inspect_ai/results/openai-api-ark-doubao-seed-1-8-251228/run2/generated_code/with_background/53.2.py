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
    # Initialize tracking arrays with initial state at time 0
    time_cor = [0.0]
    prey_evol = [prey]
    predator_evol = [predator]
    current_time = 0.0

    while True:
        # Get current population state
        current_prey = prey_evol[-1]
        current_predator = predator_evol[-1]

        # Execute one Gillespie step
        delta_t, new_prey, new_predator, event = gillespie_step(current_prey, current_predator, alpha, beta, gamma)

        # Terminate if no further events can occur
        if delta_t is None:
            break

        # Calculate time of next event
        next_time = current_time + delta_t

        # Stop simulation if next event exceeds final time T
        if next_time > T:
            break

        # Update state and tracking arrays
        current_time = next_time
        time_cor.append(current_time)
        prey_evol.append(new_prey)
        predator_evol.append(new_predator)

    # Convert lists to numpy arrays as specified
    time_cor = np.array(time_cor, dtype=float)
    prey_evol = np.array(prey_evol, dtype=float)
    predator_evol = np.array(predator_evol, dtype=float)

    # Determine the ecological event based on final state
    final_prey = prey_evol[-1]
    final_predator = predator_evol[-1]

    if final_prey > 0 and final_predator > 0:
        eco_event = "coexistence"
    elif final_prey > 0 and final_predator == 0:
        eco_event = "predator extinction"
    else:
        eco_event = "mutual extinction"

    return time_cor, prey_evol, predator_evol, eco_event
