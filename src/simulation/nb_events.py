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
    SIR_transition_rates,
    where_infections_happened_counter) :

    N_tot = my.cfg_network.N_tot
    event_size_max = my.cfg.event_size_max
    # if no max, set it to N_tot
    if event_size_max == 0 :
        event_size_max = N_tot

    # if weekend increase number of events
    if (day % 7) == 0 or (day % 7) == 1 :
        my.cfg.N_events = int(
            my.cfg.N_events * my.cfg.event_weekend_multiplier
        )  #  randomness x XXX

    # agents_in_event = List()

    agents_getting_infected_at_any_event = List()

    for _ in range(my.cfg.N_events) :

        event_size = min(
            int(10 - np.log(np.random.rand()) * my.cfg.event_size_mean),
            event_size_max,
        )
        event_duration = -np.log(np.random.rand()) * 2 / 24  # event duration in days

        event_id = np.random.randint(N_tot)

        agents_in_this_event = List()
        while len(agents_in_this_event) < event_size :
            guest = np.random.randint(N_tot)
            rho_tmp = 0.5
            epsilon_rho_tmp = 4 / 100
            if my.dist_accepted(event_id, guest, rho_tmp) or np.random.rand() < epsilon_rho_tmp :
                agents_in_this_event.append(np.uint32(guest))

        # extract all agents that are infectious and then
        for agent in agents_in_this_event :
            if my.agent_is_infectious(agent) :
                for guest in agents_in_this_event :
                    if guest != agent and my.agent_is_susceptible(guest) :
                        time = np.random.uniform(0, event_duration)
                        probability = my.infection_weight[agent] * time * my.cfg.event_beta_scaling

                        if np.random.rand() < probability :
                            if guest not in agents_getting_infected_at_any_event :
                                agents_getting_infected_at_any_event.append(np.uint32(guest))

    for agent_getting_infected_at_event in agents_getting_infected_at_any_event :

        # XXX this update was needed
        my.state[agent_getting_infected_at_event] = 0
        where_infections_happened_counter[3] += 1
        agents_in_state[0].append(np.uint32(agent_getting_infected_at_event))
        state_total_counts[0] += 1
        g.total_sum_of_state_changes += SIR_transition_rates[0]
        g.cumulative_sum_of_state_changes += SIR_transition_rates[0]

        nb_simulation.update_infection_list_for_newly_infected_agent(my, g, agent_getting_infected_at_event)