import numpy as np

from numba import njit
from numba.typed import List

import src.simulation.nb_simulation as nb_simulation

######## ##     ## ######## ##    ## ########  ######
##       ##     ## ##       ###   ##    ##    ##    ##
##       ##     ## ##       ####  ##    ##    ##
######   ##     ## ######   ## ## ##    ##     ######
##        ##   ##  ##       ##  ####    ##          ##
##         ## ##   ##       ##   ###    ##    ##    ##
########    ###    ######## ##    ##    ##     ######


@njit
def add_daily_events(
    my,
    g,
    day,
    agents_in_state,
    state_total_counts,
    stratified_infection_counts,
    where_infections_happened_counter) :

    N_tot = my.cfg_network.N_tot
    event_size_max = my.cfg.event_size_max

    # if no max, set it to N_tot
    if event_size_max == 0 :
        event_size_max = N_tot

    # Determine the number of events
    N_events = my.cfg.N_events

    # if weekend increase number of events
    if (day % 7) == 0 or (day % 7) == 1 :
        N_events *= my.cfg.event_weekend_multiplier


    # Find all agents getting infected
    agents_getting_infected_at_any_event = List()

    for _ in range(int(N_events)) :

        # Choose event type
        event_beta_scaling = my.cfg.event_beta_scaling
        if np.random.rand() < my.cfg.outdoor_indoor_event_ratio :
            event_beta_scaling *= my.cfg.outdoor_beta_scaling

        # Choose event size and duration
        event_size = int(-np.log(np.random.rand()) * my.cfg.event_size_mean)
        event_size = min(event_size, event_size_max)

        event_duration = -np.log(np.random.rand()) * 2 / 24  # event duration in days (average 2 hours)

        event_id = np.random.randint(N_tot)

        # Choose agents for the event
        agents_in_this_event = List()
        while len(agents_in_this_event) < event_size :
            guest = np.random.randint(N_tot)
            rho_tmp = my.cfg.event_rho
            epsilon_rho_tmp = 4 / 100
            if my.dist_accepted(event_id, guest, rho_tmp) or np.random.rand() < epsilon_rho_tmp :
                agents_in_this_event.append(np.uint32(guest))


        # extract all agents that are infectious
        for agent in agents_in_this_event :
            if my.agent_is_infectious(agent) :

                # and then try to infect
                for guest in agents_in_this_event :
                    if guest != agent and my.agent_is_susceptible(guest) :

                        # How long did they interact at the event?
                        time = np.random.uniform(0, event_duration)
                        probability = my.infection_weight[agent] * time * event_beta_scaling

                        # Infect
                        if np.random.rand() < probability :
                            if guest not in agents_getting_infected_at_any_event :
                                agents_getting_infected_at_any_event.append(np.uint32(guest))


    # Implement the infections from events
    for agent in agents_getting_infected_at_any_event :

        # Update the state
        my.state[agent] = 0
        agents_in_state[0].append(np.uint32(agent))
        state_total_counts[0] += 1
        g.total_sum_of_state_changes += g.SIR_transition_rates[0]
        g.cumulative_sum_of_state_changes += g.SIR_transition_rates[0]

        # Update the network
        nb_simulation.update_infection_list_for_newly_infected_agent(my, g, agent)

        # Update the counters
        where_infections_happened_counter[3] += 1
        stratified_infection_counts[my.label[agent]][my.corona_type[agent]][my.age[agent]] += 1
