import numpy as np

from numba import njit

from numba.typed import List

from src.utils import utils

from src.simulation.nb_interventions   import testing_intervention, vaccinate, apply_daily_interventions
from src.simulation.nb_interventions   import apply_symptom_testing, apply_random_testing
from src.simulation.nb_interventions   import calculate_R_True, calculate_R_True_brit, calculate_population_freedom_impact
from src.simulation.nb_events          import add_daily_events
from src.simulation.nb_helpers         import nb_random_choice, single_random_choice

#                        ..   ".                            .
#                         >UM@@@k+                  .c%@$%mu'
#                        'b@$@B$0l                   (#@@@@Br.
#                         .-jhB@n.       .``'.       Ik@8L/^  .
#                          ...jBa-     .]pB@BdI   . .m@p,. .
#                          .   z8#{.     (M@Z!    .I0Bdi  .
#                              'Q%%wcpooo#B$8**a0uXoBM1       .
#          .         .)U%#_  :[o@@$@$B@@$$$$$$@@@@$@$BZt: .m@@#}`          .
#         ",.       .rB@@Bv/#%@B@@@@B$@@$$$$$$@@B@$@@B@BBawW@BB#_         ^:'
#      '}h@@W/'       :,<q@$$$@$@@$@@@$@$$$$$$$$@@$@$$@$@$$@M[,         lZBB8X;
#      Ib@$@@@k[l    . )MB@@$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$@$@&C;     ^>U8@B@@@u.
#      "JooZccQa*B8wn-d@@@@@$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$@B@@%h/Jw*B8hzccch*h1.
#                "_|#@@$@@@@$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$B$@$@@@0{,`
#             .    `d@$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$@0`      .
#              . ../#@$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$@$r.   .
#             [W$&zh@$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$Bu?h@M)
#             ]M@%pW@@$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ahB@@Q'
#              ,;,.Y8@$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$@@r.;lI   .
#                  ^k@$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$@o] .
#               ..;+O%@@@@@@$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$@B$@$@@#ri'           .
#      'tqpx_~tq8@@@Ow@@B$@@$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$@@@B8X|M@@MdQ_++Zd0>.
#     .CB@@@@8Lr|'    ^mBB@@$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$B@BBa?     Ijp*B$B@$c.
#      `u8@@a-         .<p@@$@$@@@$$$$$$$$$$$$$$$$$$$@@B$@B$Wj^         "YBB@k~
#         ]]: .    . Iw@@BBo8@$BB@$$$$$$$$$$$$$$$$$$$@B@BBo&@@@*_      . '-[l    .
#                     <pB@h+'1wW@@$$$$$$$$$$$$$$$$$$@Bo0]I'UB@8u"      .     '
#            .           !     'q@&d*@@@@$@$@$@$@8k%@d<     ^`
#                           . .UB&v . ..ld@@O<` . ._bBq!
#                            .tB#).   ..XB@@W1 ..   ,w8z'.
#                        . :>Z%@n.    .  :;;,  .     ih@Mr!.
#                        .OB@@B@J"                   +oB$@@M).
#                         ]#@@B$W[                  'm@@%@#{^ .
#                           .   .                         .











# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # INITIAL INFECTIONS  # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@njit
def make_random_initial_infections(my, possible_agents, N, prior = None) :

    if prior is None :
        prior = np.ones(len(possible_agents)) / len(possible_agents)

    if my.cfg.weighted_random_initial_infections :
        prior_contacts = np.array([7.09189651e+00, 7.21828639e+00, 7.35063322e+00, 7.48921778e+00,
           7.63433406e+00, 7.78628991e+00, 7.94540769e+00, 8.11202496e+00,
           8.28649517e+00, 8.46918845e+00, 8.66049237e+00, 8.86081276e+00,
           9.07057458e+00, 9.29022282e+00, 9.52022345e+00, 9.76106439e+00,
           1.00132566e+01, 1.02773350e+01, 1.05538598e+01, 1.08434177e+01,
           1.11466230e+01, 1.14641189e+01, 1.17965788e+01, 1.21447081e+01,
           1.25092454e+01, 1.28909638e+01, 1.32906734e+01, 1.37092218e+01,
           1.41474972e+01, 1.46064292e+01, 1.50869914e+01, 1.55902033e+01,
           1.61171325e+01, 1.66688966e+01, 1.72466664e+01, 1.78516673e+01,
           1.84851830e+01, 1.91485573e+01, 1.98431976e+01, 2.05705774e+01,
           2.13322397e+01, 2.21298006e+01, 2.29649517e+01, 2.38394649e+01,
           2.47551954e+01, 2.57140858e+01, 2.67181702e+01, 2.77695789e+01,
           2.88705421e+01, 3.00233956e+01, 3.12305849e+01, 3.24946711e+01,
           3.38183358e+01, 3.52043869e+01, 3.66557650e+01, 3.81755489e+01,
           3.97669627e+01, 4.14333825e+01, 4.31783434e+01, 4.50055472e+01,
           4.69188701e+01, 4.89223712e+01, 5.10203005e+01, 5.32171087e+01,
           5.55174561e+01, 5.79262227e+01, 6.04485185e+01, 6.30896942e+01,
           6.58553529e+01, 6.87513617e+01, 7.17838642e+01, 7.49592935e+01,
           7.82843861e+01, 8.17661957e+01, 8.54121088e+01, 8.92298599e+01,
           9.32275478e+01, 9.74136534e+01, 1.01797057e+02, 1.06387058e+02,
           1.11193393e+02, 1.16226259e+02, 1.21496331e+02, 1.27014792e+02,
           1.32793346e+02, 1.38844253e+02, 1.45180349e+02, 1.51815076e+02,
           1.58762509e+02, 1.66037386e+02, 1.73655140e+02, 1.81631931e+02,
           1.89984681e+02, 1.98731110e+02, 2.07889773e+02, 2.17480099e+02,
           2.27522433e+02, 2.38038079e+02, 2.49049344e+02, 2.60579589e+02,
           2.72653273e+02, 2.85296010e+02, 2.98534620e+02, 3.12397188e+02,
           3.26913121e+02, 3.42113214e+02, 3.58029713e+02, 3.74696383e+02,
           3.92148580e+02, 4.10423328e+02, 4.29559396e+02, 4.49597378e+02,
           4.70579783e+02, 4.92551123e+02, 5.15558010e+02, 5.39649249e+02,
           5.64875947e+02, 5.91291622e+02, 6.18952312e+02, 6.47916695e+02,
           6.78246218e+02, 7.10005221e+02, 7.43261078e+02, 7.78084340e+02,
           8.14548879e+02, 8.52732052e+02, 8.92714861e+02, 9.34582126e+02,
           9.78422665e+02, 1.02432948e+03, 1.07239996e+03, 1.12273608e+03,
           1.17544463e+03, 1.23063741e+03, 1.28843153e+03, 1.34894957e+03,
           1.41231994e+03, 1.47867705e+03, 1.54816168e+03, 1.62092124e+03,
           1.69711008e+03, 1.77688982e+03, 1.86042971e+03, 1.94790698e+03,
           2.03950719e+03, 2.13542468e+03, 2.23586291e+03, 2.34103497e+03,
           2.45116395e+03, 2.56648350e+03, 2.68723825e+03, 2.81368437e+03,
           2.94609011e+03, 3.08473635e+03, 3.22991721e+03, 3.38194069e+03,
           3.54112929e+03, 3.70782070e+03, 3.88236856e+03, 4.06514315e+03,
           4.25653221e+03, 4.45694175e+03, 4.66679693e+03, 4.88654292e+03,
           5.11664591e+03, 5.35759404e+03, 5.60989844e+03, 5.87409437e+03,
           6.15074229e+03, 6.44042909e+03, 6.74376930e+03, 7.06140644e+03,
           7.39401435e+03, 7.74229861e+03, 8.10699809e+03, 8.48888646e+03,
           8.88877386e+03, 9.30750861e+03, 9.74597902e+03, 1.02051153e+04,
           1.06858914e+04, 1.11893272e+04, 1.17164909e+04, 1.22685006e+04,
           1.28465275e+04, 1.34517977e+04, 1.40855953e+04, 1.47492649e+04,
           1.54442143e+04, 1.61719178e+04, 1.69339191e+04, 1.77318349e+04,
           1.85673577e+04, 1.94422602e+04, 2.03583982e+04, 2.13177153e+04,
           2.23222466e+04, 2.33741232e+04, 2.44755764e+04, 2.56289429e+04])

        prob = prior_contacts[my.number_of_contacts[possible_agents]]
        prob = prior * prob / prob.sum()
    else :
        prob = prior.copy()

    return nb_random_choice(
        possible_agents,
        prob=prob,
        size=N,
        replace=False)

@njit
def make_initial_infections_in_households(my, possible_agents, N) :

    household_contacts = np.array([contact for agent in possible_agents for ith_contact, contact in enumerate(my.connections[agent]) if my.connection_type[agent][ith_contact] == 0])

    if N >= len(household_contacts) :
        return household_contacts

    else :
        return nb_random_choice(
            household_contacts,
            prob=np.ones(len(household_contacts)),
            size=N,
            replace=False)


@njit
def choose_initial_agents(my, possible_agents, N, prior) :

    #  Standard outbreak type, infecting randomly
    if my.cfg.make_random_initial_infections :

        # Choose agents randomly
        return make_random_initial_infections(my, possible_agents, N, prior)


    # Local outbreak type, infecting around a point :
    else :
        raise ValueError("Prior not implemented here")
        rho_init_local_outbreak = 0.1

        outbreak_agent = single_random_choice(possible_agents)  # this is where the outbreak starts

        initial_agents = List()
        initial_agents.append(outbreak_agent)

        while len(initial_agents) < N :
            proposed_agent = single_random_choice(possible_agents)

            if my.dist_accepted(outbreak_agent, proposed_agent, rho_init_local_outbreak) :
                if proposed_agent not in initial_agents :
                    initial_agents.append(proposed_agent)
        return np.asarray(initial_agents, dtype=np.uint32)


@njit
def find_outbreak_agent(my, possible_agents, coordinate, rho, max_tries=10_000) :
    counter = 0
    while True :
        outbreak_agent = single_random_choice(possible_agents)
        if my.dist_accepted_coordinate(outbreak_agent, coordinate, rho) :
            return outbreak_agent
        counter += 1
        if counter >= max_tries :
            raise AssertionError("Couldn't find any outbreak agent!")


@njit
def calc_E_I_distribution_linear(R_guess) :
    delta = 4 / 8 * (R_guess - 1) / (R_guess + 1)
    return np.ones(8, dtype=np.float64) - np.array([4, 3, 2, 1, -1, -2, -3, -4]) * delta



@njit
def calc_E_I_distribution(my, r_guess) :
    p_E = 4/my.cfg.lambda_E
    p_I = 4/my.cfg.lambda_I
    gen = p_E + p_I
    E_I_weight_list = np.ones(8, dtype=np.float64)
    for i in range(1,4) :
        E_I_weight_list[i :] = E_I_weight_list[i :] * r_guess**(1/my.cfg.lambda_I/(p_E*p_I))
    for i in range(4,8) :
        E_I_weight_list[i :] = E_I_weight_list[i :] * r_guess**(1/my.cfg.lambda_E/(p_E*p_I))
    E_I_weight_list = E_I_weight_list[ : :-1]
    return E_I_weight_list


@njit
def find_possible_agents(my, initial_ages_exposed, agents_in_age_group) :
    # version 2 has age groups
    if my.cfg.version >= 2 :
        possible_agents = List()
        for age_exposed in initial_ages_exposed :
            for agent in agents_in_age_group[age_exposed] :
                possible_agents.append(agent)
        possible_agents = np.asarray(possible_agents, dtype=np.uint32)
    # version 1 has no age groups
    else :
        possible_agents = np.arange(my.cfg_network.N_tot, dtype=np.uint32)

    return possible_agents


#@njit
def initialize_states(
    my,
    g,
    intervention,
    state_total_counts,
    stratified_infection_counts,
    stratified_label_map,
    agents_in_state,
    subgroup_UK_frac,
    possible_agents,
    N_init,
    R_init,
    prior_infected,
    prior_immunized,
    nts,
    verbose=False) :


    if N_init > 0 :
        agents = choose_initial_agents(my, possible_agents, N_init, prior_infected)

        #  Make initial infections
        for agent in agents :

            # Choose corona type
            if np.random.rand() < subgroup_UK_frac * my.cfg.N_init_UK_frac :
                my.corona_type[agent] = 1
                rel_beta = my.cfg.beta_UK_multiplier
            else :
                rel_beta = 1

            #weights = calc_E_I_distribution_linear(my.cfg.R_guess * rel_beta)
            weights = calc_E_I_distribution(my, my.cfg.R_guess * rel_beta)
            states = np.arange(g.N_states - 1, dtype=np.int8)
            new_state = nb_random_choice(states, weights, verbose=verbose)[0]  # E1-E4 or I1-I4, uniformly distributed
            my.state[agent] = new_state

            agents_in_state[new_state].append(np.uint32(agent))
            state_total_counts[new_state] += 1

            g.total_sum_of_state_changes += g.SIR_transition_rates[new_state]
            g.cumulative_sum_of_state_changes[new_state :] += g.SIR_transition_rates[new_state]

            if intervention.apply_interventions and intervention.apply_symptom_testing :

                for i in range(4, new_state) :

                    apply_symptom_testing(my, intervention, agent, i, 0)

                    if intervention.result_of_test[agent] == 1 :

                        # Randomize the time of the test
                        day_when_symptom_testing = np.random.rand() * (i - new_state) / (my.cfg.lambda_I)
                        click_when_symptom_testing = np.int32(day_when_symptom_testing / nts)

                        intervention.clicks_when_tested[agent]   = click_when_symptom_testing + intervention.cfg.test_delay_in_clicks[0]
                        intervention.clicks_when_isolated[agent] = click_when_symptom_testing

                        # Break the symptom testing loop
                        break


            # Moves into a infectious State
            if my.agent_is_infectious(agent) :
                for ith_contact, contact in enumerate(my.connections[agent]) :

                    # Strain specific multiplier
                    if my.corona_type[agent] == 1 :
                        g.rates[agent][ith_contact] *= my.cfg.beta_UK_multiplier

                    # update rates if contact is susceptible
                    if my.agent_is_connected(agent, ith_contact) and my.agent_is_susceptible(contact) :

                        # Set the rates
                        rate = g.rates[agent][ith_contact]
                        g.update_rates(my, +rate, agent)

                # Update the counters
                stratified_infection_counts[stratified_label_map[my.sogn[agent]]][my.corona_type[agent]][my.age[agent]] += 1

            # Make sure agent can not be re-infected
            update_infection_list_for_newly_infected_agent(my, g, agent)


    R_state = g.N_states - 1

    if R_init > 0 :

        agents = choose_initial_agents(my, possible_agents, R_init, prior_immunized)

        #  Make initial immunizations
        for agent in agents :

            # If infected, do not immunize # TODO: Discuss if this is the best way to immunize agents
            if my.state[agent] >= 0 :
                continue

            # Update the state
            my.state[agent] = R_state

            if np.random.rand() < subgroup_UK_frac * my.cfg.N_init_UK_frac :
                my.corona_type[agent] = 1

            agents_in_state[R_state].append(np.uint32(agent))

            state_total_counts[R_state] += 1

            g.total_sum_of_state_changes += g.SIR_transition_rates[R_state]
            g.cumulative_sum_of_state_changes[R_state :] += g.SIR_transition_rates[R_state]

            # Disable incomming rates
            update_infection_list_for_newly_infected_agent(my, g, agent)


@njit
def initialize_testing(my, g, intervention, nts) :

    start_click = np.int32( - g.N_infectious_states / (my.cfg.lambda_I * nts))
    print(start_click)
    # Loop over all posible clicks
    for click in range(start_click, 0) :
        print(click)
        # Implement the consequences of testing
        testing_intervention(my, g, intervention, np.int32(click*nts), click)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # PRE SIMULATION  # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@njit
def update_infection_list_for_newly_infected_agent(my, g, agent) :

    # Here we update infection lists so that newly infected cannot be infected again

    # loop over contacts of the newly infected agent in order to :
    # 1) remove newly infected agent from contact list (find_myself) by setting rate to 0
    # 2) remove rates from contacts gillespie sums (only if they are in infections state (I))
    for ith_contact, contact in enumerate(my.connections[agent]) :

        if not my.agent_is_connected(agent, ith_contact) :
            continue

        # loop over indexes of the contact to find_myself and set rate to 0
        for ith_contact_of_contact in range(my.number_of_contacts[contact]) :

            find_myself = my.connections[contact][ith_contact_of_contact]

            # check if the contact found is myself
            if find_myself == agent :

                rate = g.rates[contact][ith_contact_of_contact]

                # set rates to myself to 0 (I cannot get infected again)
                g.rates[contact][ith_contact_of_contact] = 0

                # if the contact can infect, then remove the rates from the overall gillespie accounting
                if my.agent_is_infectious(contact) :
                    g.update_rates(my, -rate, contact)

                break

@njit
def do_bug_check(
    my,
    g,
    step_number,
    day,
    continue_run,
    verbose,
    state_total_counts,
    accept,
    ra1,
    s,
    x) :

    if day > my.cfg.day_max :
        if verbose :
            print("--- day exceeded day_max ---")
        continue_run = False

    elif day > 10_000 :
        if verbose :
            print("--- day exceeded 10_000 ---")
        continue_run = False

    elif step_number > 100_000_000 :
        if verbose :
            print("=== step_number > 100_000_000 === ")
        continue_run = False

    elif (g.total_sum_infections * g.seasonality(day) + g.total_sum_of_state_changes < 0.0001) and (g.total_sum_of_state_changes + g.total_sum_infections * g.seasonality(day)  > -0.00001) :
        continue_run = False
        if verbose :
            print("Equilibrium")
            print(day, my.cfg.day_max, my.cfg.day_max > 0, day > my.cfg.day_max)

    elif state_total_counts[g.N_states - 1] > my.cfg_network.N_tot - 10 :
        if verbose :
            print("2/3 through")
        continue_run = False

    # Check for bugs
    elif not accept :
        print("\nNo Chosen rate")
        print("s : \t", s)
        print("g.total_sum_infections : \t", g.total_sum_infections * g.seasonality(day) )
        print("g.cumulative_sum_infection_rates : \t", g.cumulative_sum_infection_rates * g.seasonality(day))
        print("g.cumulative_sum_of_state_changes : \t", g.cumulative_sum_of_state_changes)
        print("x : \t", x)
        print("ra1 : \t", ra1)
        continue_run = False

    elif (g.total_sum_of_state_changes < 0) and (g.total_sum_of_state_changes > -0.001) :
        g.total_sum_of_state_changes = 0

    elif (g.total_sum_infections * g.seasonality(day)  < 0) and (g.total_sum_infections * g.seasonality(day)  > -0.001) :
        g.total_sum_infections = 0

    elif (g.total_sum_of_state_changes < 0) or (g.total_sum_infections * g.seasonality(day)  < 0) :
        print("\nNegative Problem", g.total_sum_of_state_changes, g.total_sum_infections * g.seasonality(day) )
        print("s : \t", s)
        print("g.total_sum_infections : \t", g.total_sum_infections * g.seasonality(day) )
        print("g.cumulative_sum_infection_rates : \t", g.cumulative_sum_infection_rates * g.seasonality(day))
        print("g.cumulative_sum_of_state_changes : \t", g.cumulative_sum_of_state_changes)
        print("x : \t", x)
        print("ra1 : \t", ra1)
        continue_run = False

    return continue_run

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # SIMULATION  # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


#@njit
def run_simulation(
    my,
    g,
    intervention,
    state_total_counts,
    stratified_infection_counts,
    stratified_label_map,
    stratified_vaccination_counts,
    agents_in_state,
    nts,
    verbose=False) :

    if verbose :
        print("Apply intervention", intervention.apply_interventions)

    # Define outputs
    out_time = List()                            # Sampled times
    out_state_counts = List()                    # Tne counts of the SEIR states
    out_stratified_infection_counts = List()     # The counts of infected per age group
    out_stratified_vaccination_counts = List()   # The counts of infected per age group
    out_my_state = List()

    daily_counter = 0
    day = 0
    click = 0
    step_number = 0

    real_time = 0.0


    s_counter = np.zeros(4)
    where_infections_happened_counter = np.zeros(4)

    # Check for day 0 interventions
    if intervention.apply_interventions :
        apply_daily_interventions(my, g, intervention, day, click, stratified_vaccination_counts, verbose)

        if my.cfg.start_date_offset > 0 :
            for d in range(-my.cfg.start_date_offset, 1) :
                vaccinate(my, g, intervention, d, stratified_vaccination_counts, verbose=verbose)

    # Run the simulation ################################
    continue_run = True
    while continue_run :

        s = 0

        step_number += 1
        total_sum = g.total_sum_of_state_changes + g.total_sum_infections * g.seasonality(day)

        dt = -np.log(np.random.rand()) / total_sum
        real_time += dt

        g.cumulative_sum = 0.0
        ra1 = np.random.rand()

        #######/ Here we move between infected between states
        accept = False
        if g.total_sum_of_state_changes / total_sum > ra1 :

            s = 1

            x = g.cumulative_sum_of_state_changes / total_sum
            state_now = np.searchsorted(x, ra1)
            state_after = state_now + 1

            agent = utils.numba_random_choice_list(agents_in_state[state_now])

            # We have chosen agent to move -> here we move it
            agents_in_state[state_after].append(agent)
            agents_in_state[state_now].remove(agent)

            my.state[agent] += 1

            state_total_counts[state_now]   -= 1
            state_total_counts[state_after] += 1

            g.total_sum_of_state_changes -= g.SIR_transition_rates[state_now]
            g.total_sum_of_state_changes += g.SIR_transition_rates[state_after]

            g.cumulative_sum_of_state_changes[state_now] -= g.SIR_transition_rates[state_now]
            g.cumulative_sum_of_state_changes[state_after :] += (
                g.SIR_transition_rates[state_after] - g.SIR_transition_rates[state_now]
            )

            g.cumulative_sum_infection_rates[state_now] -= g.sum_of_rates[agent]

            accept = True

            if intervention.apply_interventions and intervention.apply_symptom_testing and day >= 0 :
                apply_symptom_testing(my, intervention, agent, my.state[agent], click)

            # Moves TO infectious State from non-infectious
            if my.state[agent] == g.N_infectious_states :

                for ith_contact, contact in enumerate(my.connections[agent]) :

                    # update rates if contact is susceptible
                    if my.agent_is_connected(agent, ith_contact) and my.agent_is_susceptible(contact) :

                        if my.corona_type[agent] == 1 :
                            g.rates[agent][ith_contact] *= my.cfg.beta_UK_multiplier

                        rate = g.rates[agent][ith_contact]
                        g.update_rates(my, +rate, agent)

                # Update the counters
                stratified_infection_counts[stratified_label_map[my.sogn[agent]]][my.corona_type[agent]][my.age[agent]] += 1

            # If this moves to Recovered state
            if my.state[agent] == g.N_states - 1 :
                for ith_contact, contact in enumerate(my.connections[agent]) :
                    # update rates if contact is susceptible
                    if my.agent_is_connected(agent, ith_contact) and my.agent_is_susceptible(contact) :
                        rate = g.rates[agent][ith_contact]
                        g.update_rates(my, -rate, agent)

                # Update counters
                stratified_infection_counts[stratified_label_map[my.sogn[agent]]][my.corona_type[agent]][my.age[agent]] -= 1

        #######/ Here we infect new states
        else :
            s = 2

            x = (g.total_sum_of_state_changes + g.cumulative_sum_infection_rates * g.seasonality(day)) / total_sum
            state_now = np.searchsorted(x, ra1)
            g.cumulative_sum = (
                g.total_sum_of_state_changes + g.cumulative_sum_infection_rates[state_now - 1] * g.seasonality(day)
            ) / total_sum  # important change from [state_now] to [state_now-1]

            agent_getting_infected = -1
            for agent in agents_in_state[state_now] :

                # suggested cumulative sum
                suggested_cumulative_sum = g.cumulative_sum + g.sum_of_rates[agent] * g.seasonality(day) / total_sum

                if suggested_cumulative_sum > ra1 :
                    ith_contact = 0
                    for rate, contact in zip(g.rates[agent], my.connections[agent]) :

                        # if contact is susceptible
                        if my.agent_is_susceptible(contact) :

                            g.cumulative_sum += rate * g.seasonality(day) / total_sum

                            # here agent infect contact
                            if g.cumulative_sum > ra1 :
                                where_infections_happened_counter[my.connection_type[agent][ith_contact]] += 1
                                my.state[contact] = 0

                                my.corona_type[contact] = my.corona_type[agent]

                                agents_in_state[0].append(np.uint32(contact))
                                state_total_counts[0] += 1
                                g.total_sum_of_state_changes += g.SIR_transition_rates[0]
                                g.cumulative_sum_of_state_changes += g.SIR_transition_rates[0]
                                accept = True
                                agent_getting_infected = contact
                                break

                            ith_contact += 1
                else :
                    g.cumulative_sum = suggested_cumulative_sum

                if accept :
                    break

            if agent_getting_infected == -1 :
                print(
                    "Error! Not choosing any agent getting infected.",
                    *("\naccept :", accept),
                    *("\nagent_getting_infected : ", agent_getting_infected),
                    *("\nstep_number", step_number),
                    "\ncfg :")

                break

            # Here we update infection lists so that newly infected cannot be infected again
            update_infection_list_for_newly_infected_agent(my, g, agent_getting_infected)

        ################

        while nts * click  < real_time :

            # Apply interventions on clicks
            if intervention.apply_interventions :
                testing_intervention(my, g, intervention, day, click)

            # Advance click
            click += 1
            daily_counter += 1

            # Check if day is over
            if daily_counter >= 10 :

                # Store the daily state
                if day >= 0 and day <= my.cfg.day_max:

                    out_time.append(real_time)
                    out_state_counts.append(state_total_counts.copy())
                    out_stratified_infection_counts.append(stratified_infection_counts.copy())
                    out_stratified_vaccination_counts.append(stratified_vaccination_counts.copy())
                    out_my_state.append(my.state.copy())

                    intervention.R_true_list.append(calculate_R_True(my, g, day))
                    intervention.freedom_impact_list.append(calculate_population_freedom_impact(intervention))
                    intervention.R_true_list_brit.append(calculate_R_True_brit(my, g))


                # Print current progress
                if verbose :
                    print('--- day : ', day, ' ---')
                    print('n_infected : ', np.round(my.cfg.N_init + np.sum(where_infections_happened_counter)))
                    print('R_true : ', np.round(intervention.R_true_list[-1], 3))
                    print('freedom_impact : ', np.round(intervention.freedom_impact_list[-1], 3))
                    print('R_true_list_brit : ', np.round(intervention.R_true_list_brit[-1], 3))
                    print('Season multiplier : ', np.round(g.seasonality(day), 2))


                # Advance day
                day += 1
                daily_counter = 0

                # Apply interventions for the new day
                if intervention.apply_interventions :
                    apply_daily_interventions(my, g, intervention, day, click, stratified_vaccination_counts, verbose)

                if day > 50 :
                    x = X

                # Apply events for the new day
                if my.cfg.N_events > 0 :
                    add_daily_events(
                        my,
                        g,
                        intervention,
                        day,
                        agents_in_state,
                        state_total_counts,
                        stratified_infection_counts,
                        stratified_label_map,
                        where_infections_happened_counter)



        continue_run = do_bug_check(
            my,
            g,
            step_number,
            day,
            continue_run,
            verbose,
            state_total_counts,
            accept,
            ra1,
            s,
            x)

        s_counter[s] += 1

    if verbose :
        print('Simulation step_number, ', step_number)
        print('s_counter', s_counter)
        print('Where', where_infections_happened_counter)
        f = 5_800_000 / my.cfg_network.N_tot
        print('daily_tests', int(f * intervention.test_counter.sum() / day))
        print('daily_test_counter', [int(f * tests / day) for tests in intervention.test_counter])
        # Smitteopspringen kontakter ca. 1250 + 600 = 1850 personer pr dag.
        print('positive_test_counter', intervention.positive_test_counter)
        print('n_found', np.sum(np.array([1 for day_found in intervention.day_found_infected if day_found>=0])))
        #label_contacts, label_infected, label_people = calculate_contact_distribution_label(my, intervention)
        #print(list(label_contacts))
        #print(list(label_infected))
        #print(list(label_people))

        # frac_inf = np.zeros((2,200))
        # for agent in range(my.cfg_network.N_tot) :
        #     n_con = my.number_of_contacts[agent]
        #     frac_inf[1,n_con] +=1
        #     if my.state[agent]>=0 and my.state[agent] < 8 :
        #         frac_inf[0,n_con] +=1
        #print(frac_inf[0, :]/frac_inf[1, :])
        # print("N_daily_tests", intervention.N_daily_tests)
        # print("N_positive_tested", N_positive_tested)

    return out_time, out_state_counts, out_stratified_infection_counts, out_stratified_vaccination_counts, out_my_state, intervention
