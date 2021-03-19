import numpy as np
import numba as nb

from numba import njit

from src.simulation.nb_helpers import nb_random_choice

#### ##    ## ######## ######## ########  ##     ## ######## ##    ## ######## ####  #######  ##    ##  ######
 ##  ###   ##    ##    ##       ##     ## ##     ## ##       ###   ##    ##     ##  ##     ## ###   ## ##    ##
 ##  ####  ##    ##    ##       ##     ## ##     ## ##       ####  ##    ##     ##  ##     ## ####  ## ##
 ##  ## ## ##    ##    ######   ########  ##     ## ######   ## ## ##    ##     ##  ##     ## ## ## ##  ######
 ##  ##  ####    ##    ##       ##   ##    ##   ##  ##       ##  ####    ##     ##  ##     ## ##  ####       ##
 ##  ##   ###    ##    ##       ##    ##    ## ##   ##       ##   ###    ##     ##  ##     ## ##   ### ##    ##
#### ##    ##    ##    ######## ##     ##    ###    ######## ##    ##    ##    ####  #######  ##    ##  ######


@njit
def calculate_contact_distribution(my, contact_type) :
    contact_dist = np.zeros(100)
    for agent in range(my.cfg_network.N_tot) :
        agent_sum = 0
        for ith_contact in range(len(my.connections[agent])) :
            if my.connection_type[agent][ith_contact] == contact_type :
                agent_sum += 1
        contact_dist[agent_sum] += 1
    return contact_dist


@njit
def calculate_contact_distribution_label(my, intervention):
    label_contacts = np.zeros(intervention.N_labels)
    label_people = np.zeros(intervention.N_labels)
    label_infected = np.zeros(intervention.N_labels)
    for agent in range(my.cfg_network.N_tot) :
        label = my.label[agent]

        if my.agent_is_infectious(agent):
            label_infected[label] += 1
        label_people[label] += 1
        label_contacts[label] += len(my.connections[agent])
    return label_contacts, label_infected, label_people


@njit
def vaccinate(my, g, intervention, day, stratified_vaccination_counts, verbose=False) :

    # Loop over vaccine types
    for i in range(len(intervention.vaccination_schedule)) :

        # Check if all vaccines have been given
        if day > intervention.vaccination_schedule[i][-1] :
            continue

        # Check if any vaccines are effective yet :
        if day >= intervention.vaccination_schedule[i][0] :

            # Get the number of new effective vaccines
            N = intervention.vaccinations_per_age_group[i][day - intervention.vaccination_schedule[i][0]]

            # Determine which agents can be vaccinated
            possible_agents_to_vaccinate = np.array( [ agent
                                                    for agent in np.arange(my.cfg_network.N_tot, dtype=np.uint32)
                                                    if N[my.age[agent]] > 0 and my.vaccination_type[agent] == 0], dtype=np.uint32)

            if len(possible_agents_to_vaccinate) > 0 :

                # Compute probability for each agent being infected
                probabilities = np.array( [ N[my.age[agent]] for agent in possible_agents_to_vaccinate ] )

                # Distribute the effective vaccines among the population
                agents = nb_random_choice(possible_agents_to_vaccinate, probabilities, size = int(np.sum(N)), verbose=verbose)
                for agent in agents :

                    # pick agent if it is susceptible (in S state)
                    if my.agent_is_susceptible(agent) :
                        # "vaccinate agent"
                        if np.random.rand() < my.cfg.Intervention_vaccination_efficacies[i-1] :
                            multiply_incoming_rates(my, g, agent, np.array([0.0, 0.0, 0.0]))  # Reduce rates to zero
                            my.vaccination_type[agent] = i

                        else :
                            my.vaccination_type[agent] = -i

                    # Update counter
                    stratified_vaccination_counts[my.age[agent]] += 1


@njit
def calculate_R_True(my, g) :
    lambda_I = my.cfg.lambda_I
    rate_sum = g.total_sum_infections
    N_infected = 0
    for agent in range(my.cfg_network.N_tot) :
        if my.agent_is_infectious(agent) :
            N_infected += 1
    return rate_sum / lambda_I / np.maximum(N_infected, 1.0) * 4


@njit
def calculate_R_True_brit(my, g) :
    lambda_I = my.cfg.lambda_I
    rate_sum = 0
    N_infected = 0
    for agent in range(my.cfg_network.N_tot) :
        if my.agent_is_infectious(agent) and my.corona_type[agent] == 1 :
            N_infected += 1
            rate_sum += g.sum_of_rates[agent]
    return rate_sum / lambda_I / np.maximum(N_infected, 1.0) * 4


@njit
def calculate_population_freedom_impact(intervention) :
    return np.mean(intervention.freedom_impact)


@njit
def compute_my_cluster_coefficient(my) :
    """calculates cluster cooefficent
    (np.mean of the first output gives cluster coeff for whole network ).
    This function is somewhat slow, since it loops over all connections. Could just random sample, but here we get the exact distribution.
    """

    cluster_coefficient = np.zeros(my.cfg_network.N_tot, dtype=np.float32)
    for agent1 in range(my.cfg_network.N_tot) :
        counter = 0
        total = 0
        for j, contact in enumerate(my.connections[agent1]) :
            for k in range(j + 1, my.number_of_contacts[agent1]) :
                if contact in my.connections[my.connections[agent1][k]] :
                    counter += 1
                    break
                total += 1
        cluster_coefficient[agent1] = counter / total
    return cluster_coefficient


@njit
def initialize_kommuner(my, df_coordinates) :
    my.kommune = np.array(df_coordinates["idx"].values, dtype=np.uint8)
    kommune_counter = df_coordinates["idx"].value_counts().sort_index()
    kommune_counter = np.array(kommune_counter, dtype=np.uint32)
    return kommune_counter


@njit
def check_if_label_needs_intervention(
    my,
    intervention,
    day,
    threshold_info) :

    infected_per_label = np.zeros_like(intervention.label_counter, dtype=np.uint32)

    for agent, day_found in enumerate(intervention.day_found_infected) :
        if day_found > max(0, day - intervention.cfg.days_looking_back) :
            infected_per_label[my.label[agent]] += 1


    it = enumerate(
        zip(
            infected_per_label,
            intervention.label_counter,
            intervention.types,
        )
    )
    for i_label, (N_infected, N_inhabitants, my_intervention_type) in it :
        for ith_intervention in range(0, len(threshold_info) + 1) :
            if my_intervention_type == 0 :
                possible_interventions = [1, 2, 7]
            elif my_intervention_type == 1 :
                possible_interventions = [9001] # random integer that doesn't mean anything,
            elif my_intervention_type == 2 :
                possible_interventions = [1,7]
            elif my_intervention_type == 7 :
                possible_interventions = [9001] # random integer that doesn't mean anything,

            if N_infected / N_inhabitants > threshold_info[ith_intervention+1][0]/100_000.0 and threshold_info[0][ith_intervention] in possible_interventions :
                if intervention.verbose :
                    intervention_type_name = ["nothing","lockdown","masking","error","error","error","error","matrix_based"]
                    print(
                        *(intervention_type_name[threshold_info[0][ith_intervention]]," at label", i_label),
                        *("at day", day),
                        *("the num of infected is", N_infected),
                        *("/", N_inhabitants),
                    )

                intervention.types[i_label] = threshold_info[0][ith_intervention]
                break


@njit
def find_reverse_connection(my, agent, ith_contact) :
    contact = my.connections[agent][ith_contact]

    # loop over indexes of the contact to find_myself
    for ith_contact_of_contact, possible_agent in enumerate(my.connections[contact]) :

        # check if the contact found is myself
        if agent == possible_agent :
            return ith_contact_of_contact, contact


@njit
def open_connection(my, g, agent, ith_contact, intervention, two_way=True) :

    contact = my.connections[agent][ith_contact]

    my.connection_status[agent][ith_contact] = True
    reset_rates_of_connection(my, g, agent, ith_contact, intervention, two_way=False)

    if two_way :
        ith_contact_of_contact, _ = find_reverse_connection(my, agent, ith_contact)
        open_connection(my, g, contact, ith_contact_of_contact, intervention, two_way=False)


@njit
def close_connection(my, g, agent, ith_contact, intervention, two_way=True) :

    contact = my.connections[agent][ith_contact]

    # zero the g rates
    rate = g.rates[agent][ith_contact]
    g.rates[agent][ith_contact] = 0
    my.connection_status[agent][ith_contact] = False

    if my.agent_is_infectious(agent) and my.agent_is_susceptible(contact) :
        g.update_rates(my, -rate, agent)

    if two_way :
        ith_contact_of_contact, _ = find_reverse_connection(my, agent, ith_contact)
        close_connection(my, g, contact, ith_contact_of_contact, intervention, two_way=False)


@njit
def reset_rates_of_connection(my, g, agent, ith_contact, intervention, two_way=True) :

    if not my.agent_is_connected(agent, ith_contact) :
        return

    contact = my.connections[agent][ith_contact]

    # Compute the infection rate
    infection_rate = my.infection_weight[agent]
    infection_rate *= my.beta_connection_type[my.connection_type[agent][ith_contact]]
    #infection_rate *= my.cfg.label_betas[my.label[agent]]

    if my.corona_type[agent] == 1 :
        infection_rate *= my.cfg.beta_UK_multiplier

    # TODO: Here we should implement transmission risk for vaccinted persons

    # Reset the g.rates if agent is not susceptible or recovered
    if my.agent_is_susceptible(contact) :
        target_rate = infection_rate
    else :
        target_rate = 0

    # Compute the new rate
    rate = target_rate - g.rates[agent][ith_contact]
    g.rates[agent][ith_contact] = target_rate

    if my.agent_is_infectious(agent) and my.agent_is_susceptible(contact) :
        g.update_rates(my, +rate, agent)

    if two_way :
        ith_contact_of_contact, _ = find_reverse_connection(my, agent, ith_contact)
        reset_rates_of_connection(my, g, contact, ith_contact_of_contact, intervention, two_way=False)


@njit
def reset_rates_of_agent(my, g, agent, intervention) :

    for ith_contact in range(my.number_of_contacts[agent]) :
        reset_rates_of_connection(my, g, agent, ith_contact, intervention)


@njit
def multiply_incoming_rates(my, g, agent, rate_multiplication) :

    # loop over all of an agents contact
    for ith_contact in range(my.number_of_contacts[agent]) :

        if not my.agent_is_connected(agent, ith_contact) :
            continue

        ith_contact_of_contact, contact = find_reverse_connection(my, agent, ith_contact)

        target_rate = g.rates[contact][ith_contact_of_contact] * rate_multiplication[my.connection_type[contact][ith_contact_of_contact]]

        # Update the g.rates
        rate = target_rate - g.rates[contact][ith_contact_of_contact]
        g.rates[contact][ith_contact_of_contact] = target_rate

        # Updates to gillespie sums
        if my.agent_is_infectious(contact) and my.agent_is_susceptible(agent):
            g.update_rates(my, rate, contact)


@njit
def remove_intervention_at_label(my, g, intervention, ith_label) :
    for agent in range(my.cfg_network.N_tot) :
        if my.label[agent] == ith_label and my.restricted_status[agent] == 1 : #TODO: Only if not tested positive
            reset_rates_of_agent(my, g, agent, intervention)
            my.restricted_status[agent] = 0


@njit
def check_if_intervention_on_labels_can_be_removed(my, g, intervention, day, click,  threshold_info) :

    infected_per_label = np.zeros(intervention.N_labels, dtype=np.int32)
    for agent, day_found in enumerate(intervention.day_found_infected) :
        if day_found > day - intervention.cfg.days_looking_back :
            infected_per_label[my.label[agent]] += 1

    it = enumerate(
        zip(
            infected_per_label,
            intervention.label_counter,
            intervention.types,
        )
    )
    for i_label, (N_infected, N_inhabitants, my_intervention_type) in it :
        for ith_intervention in range(0, len(threshold_info)-1) :
            if N_infected / N_inhabitants < threshold_info[ith_intervention + 1][1]/100_000.0 and my_intervention_type == threshold_info[0][ith_intervention] :
                intervention.clicks_when_restriction_stops[i_label] = click + my.cfg.intervention_removal_delay_in_clicks


@njit
def loop_update_rates_of_contacts(
    my, g, intervention, agent, contact, rate, agent_update_rate, rate_reduction) :

    # updates to gillespie sums, if agent is infected and contact is susceptible
    if my.agent_is_infectious(agent) and my.agent_is_susceptible(contact) :
        agent_update_rate += rate

    # loop over indexes of the contact to find_myself and set rate to 0
    for ith_contact_of_contact, possible_agent in enumerate(my.connections[contact]) :

        if not my.agent_is_connected(contact, ith_contact_of_contact) :
            continue

        # check if the contact found is myself
        if agent == possible_agent :

            # update rates from contact to agent. Rate_reduction makes it depending on connection type
            c_rate = (
                g.rates[contact][ith_contact_of_contact]
                * rate_reduction[my.connection_type[contact][ith_contact_of_contact]]
            )
            intervention.freedom_impact[contact] += rate_reduction[my.connection_type[contact][ith_contact_of_contact]]/my.number_of_contacts[contact]
            g.rates[contact][ith_contact_of_contact] -= c_rate

            # updates to gillespie sums, if contact is infectious and agent is susceptible
            if my.agent_is_infectious(contact) and my.agent_is_susceptible(agent) :
                g.update_rates(my, -c_rate, contact)
            break

    return agent_update_rate


@njit
def cut_rates_of_agent(my, g, intervention, agent, rate_reduction) :

    agent_update_rate = 0.0

    # step 1 loop over all of an agents contact
    for ith_contact, contact in enumerate(my.connections[agent]) :

        # update rates from agent to contact. Rate_reduction makes it depending on connection type

        rate = g.rates[agent][ith_contact] * rate_reduction[my.connection_type[agent][ith_contact]]
        intervention.freedom_impact[contact] += rate_reduction[my.connection_type[agent][ith_contact]]/my.number_of_contacts[agent]

        g.rates[agent][ith_contact] -= rate

        agent_update_rate = loop_update_rates_of_contacts(
            my,
            g,
            intervention,
            agent,
            contact,
            rate,
            agent_update_rate,
            rate_reduction=rate_reduction,
        )

    # actually updates to gillespie sums
    g.update_rates(my, -agent_update_rate, agent)


@njit
def reduce_frac_rates_of_agent(my, g, intervention, agent, rate_reduction) :
    # rate reduction is 2 3-vectors. is used for masking interventions
    agent_update_rate = 0.0
    remove_rates = rate_reduction[0]
    reduce_rates = rate_reduction[1]

    # step 1 loop over all of an agents contact
    for ith_contact, contact in enumerate(my.connections[agent]) :

        # update rates from agent to contact. Rate_reduction makes it depending on connection type
        if np.random.rand() > remove_rates[my.connection_type[agent][ith_contact]] :
            act_rate_reduction = np.array([0, 0, 0], dtype=np.float64)
        else :
            act_rate_reduction = reduce_rates

        rate = (
            g.rates[agent][ith_contact]
            * act_rate_reduction[my.connection_type[agent][ith_contact]]
        )
        intervention.freedom_impact[agent] += act_rate_reduction[my.connection_type[agent][ith_contact]]/my.number_of_contacts[agent]
        g.rates[agent][ith_contact] -= rate

        agent_update_rate = loop_update_rates_of_contacts(
            my,
            g,
            intervention,
            agent,
            contact,
            rate,
            agent_update_rate,
            rate_reduction=act_rate_reduction,
        )

    # actually updates to gillespie sums
    g.update_rates(my, -agent_update_rate, agent)


@njit
def remove_and_reduce_rates_of_agent(my, g, intervention, agent, rate_reduction) :
    # rate reduction is 2 3-vectors. is used for lockdown interventions
    agent_update_rate = 0.0
    remove_rates = rate_reduction[0]
    reduce_rates = rate_reduction[1]

    # step 1 loop over all of an agents contact
    for ith_contact, contact in enumerate(my.connections[agent]) :

        # update rates from agent to contact. Rate_reduction makes it depending on connection type
        act_rate_reduction = reduce_rates
        if np.random.rand() < remove_rates[my.connection_type[agent][ith_contact]] :
            act_rate_reduction = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        rate = (
            g.rates[agent][ith_contact]
            * act_rate_reduction[my.connection_type[agent][ith_contact]]
        )

        g.rates[agent][ith_contact] -= rate
        intervention.freedom_impact[agent] += act_rate_reduction[my.connection_type[agent][ith_contact]]/my.number_of_contacts[agent]

        agent_update_rate = loop_update_rates_of_contacts(
            my,
            g,
            intervention,
            agent,
            contact,
            rate,
            agent_update_rate,
            rate_reduction=act_rate_reduction,
        )

    # actually updates to gillespie sums
    g.update_rates(my, -agent_update_rate, agent)


@njit
def remove_and_reduce_rates_of_agent_matrix(my, g, intervention, agent, n, label) :

    # Extract the contact matrices
    if n == 0 :
        work_matrix_previous  = my.cfg_network.work_matrix
        other_matrix_previous = my.cfg_network.other_matrix
    else :
        work_matrix_previous  = intervention.work_matrix_restrict[n-1][label]
        other_matrix_previous = intervention.other_matrix_restrict[n-1][label]

    work_matrix_max  = my.cfg_network.work_matrix
    other_matrix_max = my.cfg_network.other_matrix

    work_matrix_current  = intervention.work_matrix_restrict[n][label]
    other_matrix_current = intervention.other_matrix_restrict[n][label]

    # Step 1, determine the contacts and their connection probability
    # Get the list of restrictable contacts and the list of restricted contacts
    possible_contacts     = my.connections[agent].copy()
    current_contacts      = np.zeros(np.shape(possible_contacts), dtype=nb.boolean)

    # Here we compute the connection probability based on the contact matrix sums
    connection_probability_current  = np.ones(np.shape(possible_contacts), dtype=np.float64)
    connection_probability_previous = np.ones(np.shape(possible_contacts), dtype=np.float64)

    for ith_contact, contact in enumerate(possible_contacts) :

        # Check for active connection
        if my.agent_is_connected(agent, ith_contact) :
            current_contacts[ith_contact] = True

        # Compute connection probabilities for non-home contacts
        if not my.connection_type[agent][ith_contact] == 0 :

            if my.connection_type[agent][ith_contact] == 1 :
                sc = work_matrix_current[my.age[agent], my.age[contact]]
                sp = work_matrix_previous[my.age[agent], my.age[contact]]
                sm = work_matrix_max[my.age[agent], my.age[contact]]

            elif my.connection_type[agent][ith_contact] == 2 :
                sc = other_matrix_current[my.age[agent], my.age[contact]]
                sp = other_matrix_previous[my.age[agent], my.age[contact]]
                sm = other_matrix_max[my.age[agent], my.age[contact]]

            connection_probability_current[ith_contact]  = sc / sm
            connection_probability_previous[ith_contact] = sp / sm

    # Step 2, check if the changes should close any of the active connections
    # Store the connection weight
    sum_connection_probability = np.sum(connection_probability_current)

    # Compute the probability modifier (generalized logistic function dependent on the log ratio of number of contacts to mu)
    # For k != 1, a factor is needed to improve accuracy. Reason unknown. See notebooks: restriction_transformation_function
    # Only needed when contact distribution has a long tail.
    # k = 1   -> factor = 1.22 ish
    # k = 0.5 -> factor = 1.25 ish
    logit = 1 / (1 + (1.22 * my.cfg_network.mu / my.number_of_contacts[agent]))

    # Loop over all active (non-home) conenctions
    for ith_contact in range(my.number_of_contacts[agent]) :

        if current_contacts[ith_contact]:

            # if a connection is active, and the connection probability is lower now than before, check if this connection should be disabled
            if connection_probability_current[ith_contact] < connection_probability_previous[ith_contact] :

                # Update the connection
                p = 1 - np.sqrt(connection_probability_current[ith_contact])
                if np.random.rand() < p :
                    close_connection(my, g, agent, ith_contact, intervention)

                else :
                    connection_probability_current[ith_contact] = 1

            # if a connection is active, and the connection probability is larger now than before, redistribute that probability
            else :
                connection_probability_current[ith_contact] = 1


    # Redistribute probability
    non_active_connection_probability = np.sum(connection_probability_current[~current_contacts])
    if non_active_connection_probability > 0 :
        k = (sum_connection_probability - np.sum(connection_probability_current[current_contacts])) / non_active_connection_probability
    else :
        k = 0

    connection_probability_current[~current_contacts] *= k

    # Step 3, add new connections as needed
    # Loop over all posible (non-home) conenctions
    for ith_contact in range(my.number_of_contacts[agent]) :

        if not current_contacts[ith_contact]:

            # Update the connection
            p = 1 - np.sqrt(1 - min(1.0, connection_probability_current[ith_contact]))
            if np.random.rand() < p :
                open_connection(my, g, agent, ith_contact, intervention)


@njit
def lockdown_on_label(my, g, intervention, label, rate_reduction) :
    # lockdown on all agent with a certain label (tent or municipality, or whatever else you define). Rate reduction is two vectors of length 3. First is the fraction of [home, job, others] rates to set to 0.
    # second is the fraction of reduction of the remaining [home, job, others] rates.
    # ie : [[0,0.8,0.8],[0,0.8,0.8]] means that 80% of your contacts on job and other is set to 0, and the remaining 20% is reduced by 80%.
    # loop over all agents
    for agent in range(my.cfg_network.N_tot) :
        if my.label[agent] == label :
            my.restricted_status[agent] = 1
            remove_and_reduce_rates_of_agent(my, g, intervention, agent, rate_reduction)


@njit
def masking_on_label(my, g, intervention, label, rate_reduction) :
    # masking on all agent with a certain label (tent or municipality, or whatever else you define). Rate reduction is two vectors of length 3. First is the fraction of [home, job, others] rates to be effected by masks.
    # second is the fraction of reduction of the those [home, job, others] rates.
    # ie : [[0,0.2,0.2],[0,0.8,0.8]] means that your wear mask when around 20% of job and other contacts, and your rates to those is reduced by 80%
    # loop over all agents
    for agent in range(my.cfg_network.N_tot) :
        if my.label[agent] == label :
            my.restricted_status[agent] = 1
            reduce_frac_rates_of_agent(my, g, intervention, agent, rate_reduction)


@njit
def matrix_restriction_on_label(my, g, intervention, label, n, verbose=False) :
    # masking on all agent with a certain label (tent or municipality, or whatever else you define). Rate reduction is two vectors of length 3. First is the fraction of [home, job, others] rates to be effected by masks.
    # second is the fraction of reduction of the those [home, job, others] rates.
    # ie : [[0,0.2,0.2],[0,0.8,0.8]] means that your wear mask when around 20% of job and other contacts, and your rates to those is reduced by 80%
    # loop over all agents

    if verbose :
        prev = 0
        for agent in range(my.cfg_network.N_tot) :
            for i in range(my.number_of_contacts[agent]) :
                if my.agent_is_connected(agent, i) :
                    prev += 1

    for agent in range(my.cfg_network.N_tot) :
        if my.label[agent] == label :
            my.restricted_status[agent] = 1
            remove_and_reduce_rates_of_agent_matrix(my, g, intervention, agent, n, label)

    if verbose :
        after = 0
        for agent in range(my.cfg_network.N_tot) :
            for i in range(my.number_of_contacts[agent]) :
                if my.agent_is_connected(agent, i) :
                    after += 1

        print("--------------")
        print("Contacts before")
        print(prev)
        print("Contacts after")
        print(after)

@njit
def test_agent(my, g, intervention, agent, click) :
    # if agent is infectious and hasn't been tested before
    if my.agent_is_infectious(agent) and intervention.agent_not_found_positive(agent):
        intervention.clicks_when_tested_result[agent] = click + intervention.cfg.results_delay_in_clicks[intervention.reason_for_test[agent]]
        intervention.positive_test_counter[intervention.reason_for_test[agent]]+= 1  # count reason found infected
        # check if tracing is on
        if intervention.apply_tracing :
            # loop over contacts
            for ith_contact, contact in enumerate(my.connections[agent]) :
                if (
                    np.random.rand()
                    < intervention.cfg.tracing_rates[my.connection_type[agent][ith_contact]]
                    and intervention.clicks_when_tested[contact] == -1
                ) :
                    intervention.reason_for_test[contact] = 2
                    intervention.clicks_when_tested[contact] = (
                        click + intervention.cfg.test_delay_in_clicks[2]
                    )
                    intervention.clicks_when_isolated[contact] = click + my.cfg.tracing_delay

    # this should only trigger if they have gone into isolation after contact tracing goes out of isolation
    elif (
        my.agent_is_not_infectious(agent)
        and intervention.agent_not_found_positive(agent)
        and click > intervention.clicks_when_isolated[agent]
    ) :
        reset_rates_of_agent(my, g, agent, intervention)

    intervention.clicks_when_isolated[agent] = -1
    intervention.clicks_when_tested[agent] = -1
    intervention.reason_for_test[agent] = -1


@njit
def apply_symptom_testing(my, intervention, agent, state, click) :

    if my.agent_is_infectious(agent) :

        prob = intervention.cfg.chance_of_finding_infected[state - 4] # TODO: Fjern hardcoded 4
        randomly_selected = np.random.rand() < prob
        not_tested_before = (intervention.clicks_when_tested[agent] == -1)

        if randomly_selected and not_tested_before :

            # testing in n_clicks for symptom checking
            intervention.clicks_when_tested[agent] = click + intervention.cfg.test_delay_in_clicks[0]

            # set the reason for testing to symptoms (0)
            intervention.reason_for_test[agent] = 0


@njit
def apply_random_testing(my, intervention, click) :
    # choose N_daily_test people at random to test
    N_daily_test = int(my.cfg.daily_tests * my.cfg_network.N_tot / 5_800_000)

    agents = np.arange(my.cfg_network.N_tot, dtype=np.uint32)

    random_agents_to_be_tested = np.random.choice(agents, N_daily_test)

    intervention.clicks_when_tested[random_agents_to_be_tested] = click + intervention.cfg.test_delay_in_clicks[1]

    # specify that random test is the reason for test
    intervention.reason_for_test[random_agents_to_be_tested] = 1


@njit
def apply_interventions_on_label(my, g, intervention, day, click, verbose=False) :

    if intervention.start_interventions_by_real_incidens_rate or intervention.start_interventions_by_meassured_incidens_rate :
        threshold_info = np.array([[1, 2], [200, 100], [20, 20]]) #TODO : remove
        check_if_intervention_on_labels_can_be_removed(my, g, intervention, day, click, threshold_info)

        for ith_label, clicks_when_restriction_stops in enumerate(intervention.clicks_when_restriction_stops) :
            if clicks_when_restriction_stops == click :
                remove_intervention_at_label(my, g, intervention, ith_label)
                intervention.clicks_when_restriction_stops[ith_label] = -1
                intervention_type_n = intervention.types[ith_label]
                intervention.types[ith_label] = 0
                intervention.started_as[ith_label] = 0
                if intervention.verbose :
                    intervention_type_name = ['nothing', 'lockdown', 'masking', 'error', 'error', 'error', 'error', 'matrix_based']
                    print(
                        *('remove ', intervention_type_name[intervention_type_n], ' at num of infected', ith_label),
                        *('at day', day)
                    )

        check_if_label_needs_intervention(
            my,
            intervention,
            day,
            threshold_info,
        )


        for ith_label, intervention_type in enumerate(intervention.types) :
            if intervention_type in intervention.cfg.threshold_interventions_to_apply :
                intervention_has_not_been_applied = intervention.started_as[ith_label] == 0

                apply_lockdown = intervention_type == 1
                if apply_lockdown and intervention_has_not_been_applied :
                    intervention.started_as[ith_label] = 1
                    lockdown_on_label(
                        my,
                        g,
                        intervention,
                        label=ith_label,
                        rate_reduction=intervention.cfg.list_of_threshold_interventions_effects[0],
                    )

                apply_masking = intervention_type == 2
                if apply_masking and intervention_has_not_been_applied :
                    intervention.started_as[ith_label] = 2
                    masking_on_label(
                        my,
                        g,
                        intervention,
                        label=ith_label,
                        rate_reduction=intervention.cfg.list_of_threshold_interventions_effects[0],
                    )

                apply_matrix_restriction = intervention_type == 7
                if apply_matrix_restriction and intervention_has_not_been_applied :
                    intervention.started_as[ith_label] = 7
                    matrix_restriction_on_label(
                        my,
                        g,
                        intervention,
                        ith_label,
                        0, #TODO: if different matrices do some fixing
                        verbose=verbose
                    )

    elif intervention.start_interventions_by_day :
        if day in list(intervention.cfg.restriction_thresholds) :
            for i, intervention_date in enumerate(intervention.cfg.restriction_thresholds) :
                if day == intervention_date :
                    if i % 2 == 0 :
                        # just looping over all labels. intervention type is not necesary with intervention by day
                        for ith_label, intervention_type in enumerate(intervention.types) :

                            # if lockdown
                            if intervention.cfg.threshold_interventions_to_apply[int(i/2)] == 1 :

                                if verbose :
                                    print('Intervention type : lockdown')

                                lockdown_on_label(
                                    my,
                                    g,
                                    intervention,
                                    label=ith_label,
                                    rate_reduction=intervention.cfg.list_of_threshold_interventions_effects[int(i/2)]
                                )

                            # if masking
                            if intervention.cfg.threshold_interventions_to_apply[int(i/2)] == 2 :
                                if verbose :
                                    print('Intervention type : masks')

                                masking_on_label(
                                    my,
                                    g,
                                    intervention,
                                    label=ith_label,
                                    rate_reduction=intervention.cfg.list_of_threshold_interventions_effects[int(i/2)]
                                )

                            # if matrix restriction
                            if intervention.cfg.threshold_interventions_to_apply[int(i/2)] == 3 :

                                if verbose :
                                    if intervention.N_labels > 1 :
                                        print('Intervention type : matrix restriction, name: ', intervention.cfg.Intervention_contact_matrices_name[int(i/2)] + '_label_' + str(ith_label) )
                                    else :
                                        print('Intervention type : matrix restriction, name: ', intervention.cfg.Intervention_contact_matrices_name[int(i/2)])

                                matrix_restriction_on_label(
                                    my,
                                    g,
                                    intervention,
                                    ith_label,
                                    int(i/2),
                                    verbose=verbose
                                )

                    else :
                        for ith_label, intervention_type in enumerate(intervention.types) :

                            # Matrix restrictions are not removed # TODO: Remove if no restrictions follow
                            if intervention.cfg.threshold_interventions_to_apply[int(i/2)] == 3 :
                                continue

                            if verbose :
                                print('Intervention removed')

                            remove_intervention_at_label(my, g, intervention, ith_label)


@njit
def test_tagged_agents(my, g, intervention, day, click) :

    # test everybody whose counter say we should test
    for agent in range(my.cfg_network.N_tot) :

        # testing everybody who should be tested
        if intervention.clicks_when_tested[agent] == click:
            test_agent(my, g, intervention, agent, click)

        if intervention.clicks_when_isolated[agent] == click and intervention.apply_isolation :
            cut_rates_of_agent(
                my,
                g,
                intervention,
                agent,
                rate_reduction=intervention.cfg.isolation_rate_reduction)

        # getting results for people
        if intervention.clicks_when_tested_result[agent] == click :

            intervention.clicks_when_tested_result[agent] = -1
            intervention.day_found_infected[agent] = day

            if intervention.apply_isolation :
                cut_rates_of_agent(
                    my,
                    g,
                    intervention,
                    agent,
                    rate_reduction=intervention.cfg.isolation_rate_reduction)