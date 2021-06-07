import numpy as np
import numba as nb

from numba import njit

from src.simulation.nb_helpers import nb_random_choice
from src.utils                 import utils

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
    label_contacts = np.zeros(intervention.N_incidence_labels)
    label_people = np.zeros(intervention.N_incidence_labels)
    label_infected = np.zeros(intervention.N_incidence_labels)
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
        if day > intervention.vaccination_schedule[i][1] :
            continue

        # Check if any vaccines are effective yet :
        if day >= intervention.vaccination_schedule[i][0] :

            # Get the number of new effective vaccines
            vaccines_to_give = intervention.vaccinations_per_age_group[i][day - intervention.vaccination_schedule[i][0]]

            # Loop over age groups and vaccinate
            for age_group, N_in_age_group in enumerate(vaccines_to_give) :

                # Skip empty age groups
                if N_in_age_group == 0 :
                    continue

                # Determine which agents can be vaccinated
                possible_agents_to_vaccinate = np.array( [ agent for agent in np.arange(my.cfg_network.N_tot, dtype=np.uint32)
                                                            if my.age[agent] == age_group and my.agent_is_not_vaccinated(agent)])

                if len(possible_agents_to_vaccinate) > 0 :

                    # Distribute the effective vaccines among the age_group
                    if N_in_age_group >= len(possible_agents_to_vaccinate) :
                        agents = possible_agents_to_vaccinate
                    else :
                        agents = nb_random_choice(possible_agents_to_vaccinate, np.ones_like(possible_agents_to_vaccinate), size = N_in_age_group, verbose=verbose)

                    # Implement the vaccination
                    for agent in agents :

                        # "vaccinate agent"
                        if np.random.rand() < my.cfg.Intervention_vaccination_efficacies[i] :
                            multiply_incoming_rates_of_agent(my, g, agent)  # Reduce rates to zero
                            my.vaccination_type[agent] =   i+1

                        else :
                            my.vaccination_type[agent] = -(i+1)

                        # Vaccines always reduces transmission risk by half
                        multiply_outgoing_rates_of_agent(my, g, agent, rate_multiplication=([0.5, 0.5, 0.5, 0.5]))

                        # No longer willing to test
                        my.testing_probability[agent] = 0.0

                        # Update counter
                        stratified_vaccination_counts[my.age[agent]] += 1


@njit
def calculate_R_True(my, g, day) :
    lambda_I = my.cfg.lambda_I
    rate_sum = g.total_sum_infections * g.seasonality(day)
    N_infected = 0
    for agent in range(my.cfg_network.N_tot) :
        if my.agent_is_infectious(agent) :
            N_infected += 1
    return rate_sum / lambda_I / np.maximum(N_infected, 1.0) * 4


@njit
def calculate_R_True_brit(my, g, day) :
    lambda_I = my.cfg.lambda_I
    rate_sum = 0
    N_infected = 0
    for agent in range(my.cfg_network.N_tot) :
        if my.agent_is_infectious(agent) and my.corona_type[agent] == 1 :
            N_infected += 1
            rate_sum += g.sum_of_rates[agent] * g.seasonality(day)
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

    # Set the flag
    my.connection_status[agent][ith_contact] = True

    # Updates to gillespie sums
    if my.agent_is_infectious(agent) and my.agent_is_susceptible(contact):
        g.update_rates(my, g.rates[agent][ith_contact], agent)

    if two_way :
        ith_contact_of_contact, _ = find_reverse_connection(my, agent, ith_contact)
        open_connection(my, g, contact, ith_contact_of_contact, intervention, two_way=False)


@njit
def close_connection(my, g, agent, ith_contact, intervention, two_way=True) :

    if not my.agent_is_connected(agent, ith_contact) :
        return

    contact = my.connections[agent][ith_contact]

    # Set the flag
    my.connection_status[agent][ith_contact] = False

    # Updates to gillespie sums
    if my.agent_is_infectious(agent) and my.agent_is_susceptible(contact):
        g.update_rates(my, -g.rates[agent][ith_contact], agent)

    if two_way :
        ith_contact_of_contact, _ = find_reverse_connection(my, agent, ith_contact)
        close_connection(my, g, contact, ith_contact_of_contact, intervention, two_way=False)


@njit
def reset_rates_of_connection(my, g, agent, ith_contact, intervention, two_way=True) :

    contact = my.connections[agent][ith_contact]

    # Compute the infection rate

    # If contact is not susceptible then rate is zero
    if not my.agent_is_susceptible(contact) :
        infection_rate = 0

    else :  # Agent is susceptiible and rates should be re-computed
        infection_rate = my.infection_weight[agent]
        infection_rate *= my.beta_connection_type[my.connection_type[agent][ith_contact]]

        if my.corona_type[agent] == 1 :
            infection_rate *= my.cfg.beta_UK_multiplier

        # Account for vaccines
        if my.agent_is_vaccinated(agent) :
            infection_rate *= 0.5


        # Account for self-isolation
        if intervention.isolated[agent] or intervention.isolated[contact] :
            infection_rate *= intervention.cfg.isolation_rate_multiplier[my.connection_type[agent][ith_contact]]


    # Store the current rate
    old_rate = g.rates[agent][ith_contact]

    # Reset the rate
    g.rates[agent][ith_contact] = infection_rate

    # Update the gillespie rates
    if my.agent_is_connected(agent, ith_contact) and my.agent_is_infectious(agent) and my.agent_is_susceptible(contact) :

        # Reset the g.rates if agent is not susceptible or recovered
        rate = infection_rate - old_rate
        g.update_rates(my, rate, agent)


    # Update the connection the other way
    if two_way :
        ith_contact_of_contact, _ = find_reverse_connection(my, agent, ith_contact)
        reset_rates_of_connection(my, g, contact, ith_contact_of_contact, intervention, two_way=False)


@njit
def reset_rates_of_agent(my, g, agent, intervention) :

    for ith_contact in range(my.number_of_contacts[agent]) :
        reset_rates_of_connection(my, g, agent, ith_contact, intervention)


@njit
def multiply_incoming_rates_of_agent(my, g, agent, rate_multiplication = np.array([0.0, 0.0, 0.0, 0.0])) :

    # loop over all of an agents contact
    for ith_contact in range(my.number_of_contacts[agent]) :

        ith_contact_of_contact, contact = find_reverse_connection(my, agent, ith_contact)

        target_rate = g.rates[contact][ith_contact_of_contact] * rate_multiplication[my.connection_type[contact][ith_contact_of_contact]]

        # Update the g.rates
        rate = target_rate - g.rates[contact][ith_contact_of_contact]
        g.rates[contact][ith_contact_of_contact] = target_rate

        # Updates to gillespie sums
        if my.agent_is_connected(contact, ith_contact_of_contact) and my.agent_is_infectious(contact) and my.agent_is_susceptible(agent):
            g.update_rates(my, rate, contact)

@njit
def multiply_outgoing_rates_of_agent(my, g, agent, rate_multiplication = np.array([0.0, 0.0, 0.0, 0.0])) :

    # loop over all of an agents contact
    for ith_contact, contact in enumerate(my.connections[agent]) :

        target_rate = g.rates[agent][ith_contact] * rate_multiplication[my.connection_type[agent][ith_contact]]

        # Update the g.rates
        rate = target_rate - g.rates[agent][ith_contact]
        g.rates[agent][ith_contact] = target_rate

        # Updates to gillespie sums
        if my.agent_is_connected(agent, ith_contact) and my.agent_is_infectious(agent) and my.agent_is_susceptible(contact):
            g.update_rates(my, rate, agent)

@njit
def multiply_rates_of_agent(my, g, agent, rate_multiplication) :
    multiply_outgoing_rates_of_agent(my, g, agent, rate_multiplication)
    multiply_incoming_rates_of_agent(my, g, agent, rate_multiplication)



@njit
def remove_intervention_at_sogn(my, g, intervention, ith_sogn) :
    for agent in range(my.cfg_network.N_tot) :
        if my.sogn[agent] == ith_sogn and my.restricted_status[agent] > 0 :

            if intervention.apply_increased_testing :
                my.testing_probability[agent] = my.testing_probability[agent] / (my.restricted_status[agent] + 1)

            if intervention.apply_lockdowns :
                reset_rates_of_agent(my, g, agent, intervention)
                my.restricted_status[agent] = 0


@njit
def add_intervention_at_sogn(my, g, intervention, sogn, ith_intervention) :
    # lockdown on all agent with a certain sogn
    # Rate reduction is a vectors of length 4.
    # The fraction of reduction of the remaining [home, job, school, others] rates.

    # loop over all agents
    for agent in range(my.cfg_network.N_tot) :
        if my.sogn[agent] == sogn :

            if intervention.apply_lockdowns :
                my.restricted_status[agent] = ith_intervention
                multiply_rates_of_agent(my, g, agent, intervention.incidence_intervention_effect[ith_intervention])

            if intervention.apply_increased_testing :
                my.testing_probability[agent] = (ith_intervention + 1) * my.testing_probability[agent]


@njit
def check_status_for_incidence_interventions(my, g, intervention, day, click) :

    # Loop over all interventions and check if condition still applies
    # if yes, write to all sogn within label of intervention

    required_interventions_at_sogn, incidence_metrics = check_incidence_against_tresholds(my, intervention, day, click)

    # Loop over sogne to remove restricitons
    for sogn, required_interventions in enumerate(required_interventions_at_sogn) :

        # If required intervntions do not match the current interventions, tag sogn for updates
        if not required_interventions == intervention.types[sogn] :

            # Store the new restriction
            intervention.types[sogn] = required_interventions

            # Set the timing of the intervention removal
            if required_interventions == 0 :
                delay = my.cfg.intervention_removal_delay_in_clicks
            else :
                delay = 0

            intervention.clicks_when_restriction_changes[sogn] = click + delay

    return incidence_metrics

@njit
def check_incidence_against_tresholds(my, intervention, day, click) :

    days_looking_back = intervention.cfg.days_looking_back

    # Incidence is computed on pcr tests
    pcr_to_total_tests_ratio = np.zeros(days_looking_back, dtype=np.float32)

    for d in range(days_looking_back) :
        if day-d >= 0 :
            pcr_to_total_tests_ratio[d] = intervention.pcr_to_total_tests_ratio[day-d]
        else :
            pcr_to_total_tests_ratio[d] = pcr_to_total_tests_ratio[d-1]

    # Loop over all interventions and check if condition applies

    # Arrays to encode information about which sogne is above the on and off tresholds
    intervention_at_sogn  = np.zeros_like(intervention.types, dtype=np.int8)

    # Loop over possible interventions
    for ith_intervention, incidence_label in enumerate(intervention.incidence_labels) :

        # Determine the number of (found) infected per label
        infected_per_label = np.zeros((intervention.N_incidence_labels[incidence_label], intervention.cfg.days_looking_back), dtype=np.float32)
        tests_per_label    = np.zeros((intervention.N_incidence_labels[incidence_label], intervention.cfg.days_looking_back), dtype=np.float32)

        for agent, (day_found, click_tested) in enumerate(zip(intervention.day_found_infected, intervention.clicks_when_tested_result)) :
            if day_found > day - days_looking_back :

                if incidence_label.lower() == 'sogn' :
                    infected_per_label[my.sogn[agent]][day-day_found]    += 1.0
                elif incidence_label.lower() == 'kommune' :
                    infected_per_label[my.kommune[agent]][day-day_found] += 1.0
                else :
                    infected_per_label[intervention.incidence_label_map[incidence_label][my.sogn[agent]]][day-day_found]  += 1.0

            if click_tested > click - intervention.clicks_looking_back :
                I = int((click - click_tested) * intervention.nts)   # Convert click to day

                if incidence_label.lower() == 'sogn' :
                    tests_per_label[my.sogn[agent]][I]    += 1.0
                elif incidence_label.lower() == 'kommune' :
                    tests_per_label[my.kommune[agent]][I] += 1.0
                else :
                    tests_per_label[intervention.incidence_label_map[incidence_label][my.sogn[agent]]][I] += 1.0


        # Loop over labels
        for ith_label, (N_tests, N_infected, N_inhabitants) in enumerate(zip(tests_per_label, infected_per_label, intervention.agents_per_incidence_label[incidence_label])) :

            if np.sum(N_tests) < 0.5 :
                continue

            # Discount AG tests
            N_tests *= pcr_to_total_tests_ratio

            # Adjust for the number of tests
            if intervention.use_only_raw_incidence or incidence_label.lower() == 'sogn' :

                # Compute the incidence on the label
                incidence = np.sum(N_infected) / (N_inhabitants / intervention.cfg.incidence_reference)

            else :

                # Scaled positives
                infected_per_label_adjusted = N_infected * (intervention.cfg.test_reference * N_inhabitants / N_tests) ** intervention.cfg.testing_exponent

                # Add the scaled positives for the week
                incidence = np.nansum(infected_per_label_adjusted) / (N_inhabitants / intervention.cfg.incidence_reference)



            # Store the incidence
            if incidence_label.lower() == 'kommune' :
                intervention.incidence_per_kommune[ith_label] = incidence


            # Check against infection treshold
            if np.sum(N_infected) <= intervention.infection_threshold[ith_intervention] :
                continue

            # Check against percentage tresholds
            if np.sum(N_infected) / np.sum(N_tests) <= intervention.percentage_threshold[ith_intervention] :
                continue

            # Check incidence against incidence treshold
            if incidence > intervention.incidence_threshold[ith_intervention] :

                # Loop over parishes on label
                for sogn in intervention.inverse_incidence_label_map[incidence_label][np.uint16(ith_label)] :
                    intervention_at_sogn[sogn] += 2**ith_intervention    # Encode as binary flags


    incidence_metrics = np.array([np.quantile(intervention.incidence_per_kommune, 0.1), np.quantile(intervention.incidence_per_kommune, 0.25), np.median(intervention.incidence_per_kommune), np.quantile(intervention.incidence_per_kommune, 0.75), np.quantile(intervention.incidence_per_kommune, 0.9), np.sum(intervention_at_sogn > 1)])

    return intervention_at_sogn, incidence_metrics

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
def reduce_rates_of_agent(my, g, intervention, agent, rate_multiplier) :
    # rate reduction is a 3-vector. is used for lockdown interventions
    agent_update_rate = 0.0

    # step 1 loop over all of an agents contact
    for ith_contact, contact in enumerate(my.connections[agent]) :

        # update rates from agent to contact. Rate_reduction makes it depending on connection type
        if np.random.rand() < remove_rates[my.connection_type[agent][ith_contact]] :
            act_rate_reduction = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        rate = (
            g.rates[agent][ith_contact]
            * rate_multiplier[my.connection_type[agent][ith_contact]]
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

    if my.number_of_contacts[agent] == 0 :
        return

    # Extract the contact matrices
    if n == 0 :
        work_matrix_previous   = my.cfg_network.work_matrix
        school_matrix_previous = my.cfg_network.school_matrix
        other_matrix_previous  = my.cfg_network.other_matrix
    else :
        work_matrix_previous   = intervention.work_matrix_restrict[n-1][label]
        school_matrix_previous = intervention.school_matrix_restrict[n-1][label]
        other_matrix_previous  = intervention.other_matrix_restrict[n-1][label]

    work_matrix_max   = my.cfg_network.work_matrix
    school_matrix_max = my.cfg_network.school_matrix
    other_matrix_max  = my.cfg_network.other_matrix

    work_matrix_current   = intervention.work_matrix_restrict[n][label]
    school_matrix_current = intervention.school_matrix_restrict[n][label]
    other_matrix_current  = intervention.other_matrix_restrict[n][label]

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
                sc = school_matrix_current[my.age[agent], my.age[contact]]
                sp = school_matrix_previous[my.age[agent], my.age[contact]]
                sm = school_matrix_max[my.age[agent], my.age[contact]]

            elif my.connection_type[agent][ith_contact] == 3 :
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

                # Naiive probability connection should be open
                p = 1 - connection_probability_current[ith_contact]

                # Transform p through a generalized logistics function
                p = logit ** (-np.log(p) / np.log(2))

                # Account for two way connection
                p = 1 - np.sqrt(1 - p)

                # Try to close connections
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

            # Naiive probability connection should be closed
            p = 1 - min(1.0, connection_probability_current[ith_contact])

            # Transform p through a generalized logistics function
            p = logit ** (-np.log(p) / np.log(2))

            # Account for two way connection
            p = 1 - np.sqrt(p)

            # Try to close connections
            if np.random.rand() < p :
                open_connection(my, g, agent, ith_contact, intervention)




@njit
def masking_on_label(my, g, intervention, label, rate_reduction) :
    # masking on all agent with a certain label (tent or municipality, or whatever else you define). Rate reduction is two vectors of length 3. First is the fraction of [home, job, others] rates to be effected by masks.
    # second is the fraction of reduction of the those [home, job, others] rates.
    # ie : [[0,0.2,0.2],[0,0.8,0.8]] means that your wear mask when around 20% of job and other contacts, and your rates to those is reduced by 80%
    # loop over all agents
    for agent in range(my.cfg_network.N_tot) :
        if intervention.incidence_label_map[my.sogn[agent]] == label :
            #my.restricted_status[agent] = 1
            reduce_frac_rates_of_agent(my, g, intervention, agent, rate_reduction)


@njit
def matrix_restriction_on_label(my, g, intervention, label, n, verbose=False) :
    # masking on all agent with a certain label (tent or municipality, or whatever else you define). Rate reduction is two vectors of length 3. First is the fraction of [home, job, others] rates to be effected by masks.
    # second is the fraction of reduction of the those [home, job, others] rates.
    # ie : [[0,0.2,0.2],[0,0.8,0.8]] means that your wear mask when around 20% of job and other contacts, and your rates to those is reduced by 80%
    # loop over all agents

    for agent in range(my.cfg_network.N_tot) :
        if intervention.matrix_label_map[my.sogn[agent]] == label :
            remove_and_reduce_rates_of_agent_matrix(my, g, intervention, agent, n, label)

@njit
def test_agent(my, g, intervention, agent, click) :

    # Set the time of result
    intervention.clicks_when_tested_result[agent] = click + intervention.cfg.results_delay_in_clicks[intervention.reason_for_test[agent]]

    # Count the tests
    intervention.test_counter[intervention.reason_for_test[agent]] += 1

    if my.agent_is_infectious(agent) :
        intervention.result_of_test[agent] = 1
    else :
        intervention.result_of_test[agent] = 0

@njit
def check_test_results(my, g, intervention, agent, day, click, stratified_positive, daily_tests) :

    daily_tests += 1

    # If agent receives positive test result
    if intervention.result_of_test[agent] == 1 :

        # Store the date found
        intervention.day_found_infected[agent] = day

        # Go into self-isolation
        intervention.clicks_when_isolated[agent] = click

        # Count the tests
        intervention.positive_test_counter[intervention.reason_for_test[agent]] += 1

        stratified_positive[intervention.stratified_label_map[my.sogn[agent]]][my.corona_type[agent]][my.age[agent]] += 1


        # Check if tracing is on
        if intervention.apply_tracing :

            # loop over contacts
            for ith_contact, contact in enumerate(my.connections[agent]) :

                # Must be connected
                if not my.agent_is_connected(agent, ith_contact) :
                    continue

                # Not all will be traced
                if np.random.rand() < intervention.cfg.tracing_rates[my.connection_type[agent][ith_contact]] : # * my.testing_probability[contact] :

                    test_click = click + my.cfg.tracing_delay + intervention.cfg.test_delay_in_clicks[2]

                    # Book new test
                    intervention.reason_for_test[contact] = 2

                    intervention.clicks_when_tested[contact] = test_click

                    # Isolate while waiting
                    intervention.clicks_when_isolated[contact] = click + my.cfg.tracing_delay

                    # No longer willing to test
                    my.testing_probability[contact] = 0.0


    else : # They recieve negative test result

        # Again willing to test
        my.testing_probability[agent] = my.cfg.testing_penetration[my.age[agent]]

        # Check for lockdown
        if intervention.apply_increased_testing and my.restricted_status[agent] > 0 :

            my.testing_probability[agent] = (my.restricted_status[agent] + 1) * my.testing_probability[agent]

        intervention.isolated[agent] = False
        reset_rates_of_agent(my, g, agent, intervention)


@njit
def apply_symptom_testing(my, intervention, agent, state, click) :

    # Infectious agents may test due to symptopns
    if my.agent_is_infectious(agent) :

        # Only thest with probabaility
        if np.random.rand() < intervention.cfg.chance_of_finding_infected[state - 4] * my.testing_probability[agent] :   # TODO: Fjern hardcoded 4

            test_click = click + intervention.cfg.test_delay_in_clicks[0]

            # Testing in n_clicks for symptom checking
            intervention.clicks_when_tested[agent] = test_click
            intervention.result_of_test[agent]     = 1

            # Set the reason for testing to symptoms (0)
            intervention.reason_for_test[agent] = 0

            # Isolate while waiting
            intervention.clicks_when_isolated[agent] = click

            # No longer willing to test
            my.testing_probability[agent] = 0.0

@njit
def apply_random_testing(my, intervention, day, click) :

    # choose N_daily_test people at random to test
    agents = np.arange(my.cfg_network.N_tot, dtype=np.uint32)

    # Choose the agents
    # local_incidence = intervention.incidence_per_kommune[my.kommune]
    random_agents_to_be_tested = np.random.uniform(0, 1, size=my.cfg_network.N_tot) < my.testing_probability * intervention.daily_test_probability[day]

    # Book test
    intervention.clicks_when_tested[random_agents_to_be_tested] = click + intervention.cfg.test_delay_in_clicks[1] + np.random.randint(0, high=int(np.round(1 / intervention.nts)), size=np.sum(random_agents_to_be_tested))

    # specify that random test is the reason for test
    intervention.reason_for_test[random_agents_to_be_tested] = 1

    # No longer willing to test
    my.testing_probability[random_agents_to_be_tested] = 0.0


    return






@njit
def apply_interventions_on_label(my, g, intervention, day, click, verbose = False) :

    if intervention.start_interventions_by_incidence :

        # Loop over sogne to update restrictions
        for ith_sogn, clicks_when_restriction_changes in enumerate(intervention.clicks_when_restriction_changes) :

            if clicks_when_restriction_changes == click :

                # Remove interventions
                remove_intervention_at_sogn(my, g, intervention, ith_sogn)

                # .. and re-apply if needed
                if intervention.types[ith_sogn] > 0 :
                    ith_intervention = -1
                    for ith_intervention in utils.decode_binary_flags(intervention.types[ith_sogn]) : # Using only the last entry in iterator
                        pass

                    add_intervention_at_sogn(my, g, intervention, ith_sogn, ith_intervention)



    if intervention.start_interventions_by_day :
        if day in list(intervention.cfg.planned_restriction_dates) :
            for i, intervention_date in enumerate(intervention.cfg.planned_restriction_dates) :
                if day == intervention_date :

                    # just looping over all labels. intervention type is not necesary with intervention by day
                    for ith_label in range(my.N_sogne) :

                        # if matrix restriction
                        if intervention.cfg.planned_restriction_types[i] == 1 :

                            if ith_label >= intervention.N_matrix_labels :
                                break

                            k = np.sum(intervention.cfg.planned_restriction_types[:i] == 1)

                            if verbose :
                                if intervention.N_matrix_labels > 1 :
                                    print('Intervention type : matrix restriction, name:', intervention.cfg.Intervention_contact_matrices_name[k] + '_label_' + str(ith_label) )
                                else :
                                    print('Intervention type : matrix restriction, name:', intervention.cfg.Intervention_contact_matrices_name[k])

                            if verbose and ith_label == 0 :
                                prev = 0
                                for agent in range(my.cfg_network.N_tot) :
                                    for ith_contact in range(my.number_of_contacts[agent]) :
                                        if my.agent_is_connected(agent, ith_contact) :
                                            prev += 1

                            matrix_restriction_on_label(my, g, intervention, ith_label, k, verbose = verbose)

                            if verbose and ith_label == intervention.N_matrix_labels - 1:
                                after = 0
                                for agent in range(my.cfg_network.N_tot) :
                                    for ith_contact in range(my.number_of_contacts[agent]) :
                                        if my.agent_is_connected(agent, ith_contact) :
                                            after += 1

                                print("--------------")
                                print("Contacts before")
                                print(prev)
                                print("Contacts after")
                                print(after)



                        # if event restrictions
                        elif intervention.cfg.planned_restriction_types[i] == 2 :

                            if ith_label >= 1 :
                                break

                            k = np.sum(intervention.cfg.planned_restriction_types[:i] == 2) + 1

                            if verbose :
                                print('Intervention type : event restriction:', intervention.cfg.event_size_max[k])

                            intervention.event_size_max = intervention.cfg.event_size_max[k]


                        # if incidence restrictions
                        elif intervention.cfg.planned_restriction_types[i] == 3 :

                            if ith_label >= 1 :
                                break

                            k = np.sum(intervention.cfg.planned_restriction_types[:i] == 3) + 1

                            if verbose :
                                print('Intervention type : incidence restriction:', intervention.cfg.incidence_labels[k])

                            intervention.incidence_labels              = intervention.cfg.incidence_labels[k]
                            intervention.incidence_threshold           = intervention.cfg.incidence_threshold[k]
                            intervention.infection_threshold           = intervention.cfg.infection_threshold[k]
                            intervention.percentage_threshold          = intervention.cfg.percentage_threshold[k]
                            intervention.incidence_intervention_effect = intervention.cfg.incidence_intervention_effect[k]


@njit
def testing_intervention(my, g, intervention, day, click, stratified_positive, daily_tests) :

    # test everybody whose counter say we should test
    for agent in range(my.cfg_network.N_tot) :

        # testing everybody who should be tested
        if intervention.clicks_when_tested[agent] == click:
            test_agent(my, g, intervention, agent, click)

        # check for test results
        if intervention.clicks_when_tested_result[agent] == click :
            check_test_results(my, g, intervention, agent, day, click, stratified_positive, daily_tests)

        # check for isolation
        if intervention.apply_isolation and intervention.clicks_when_isolated[agent] == click and not intervention.isolated[agent] :
            intervention.isolated[agent] = True
            multiply_rates_of_agent(my, g, agent, rate_multiplication = intervention.cfg.isolation_rate_multiplier)


@njit
def apply_daily_interventions(my, g, intervention, day, click, stratified_vaccination_counts, verbose) :
    if intervention.apply_interventions_on_label and day >= 0 :
        apply_interventions_on_label(my, g, intervention, day, click, verbose)

    if intervention.apply_random_testing :
        apply_random_testing(my, intervention, day, click)

    if intervention.apply_vaccinations :
        vaccinate(my, g, intervention, day, stratified_vaccination_counts, verbose = verbose)