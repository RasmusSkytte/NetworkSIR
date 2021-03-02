import numpy as np

import numba as nb
from numba import njit
from numba.typed import List

from src.utils import utils

from src.simulation.nb_jitclass import *
from src.simulation.nb_network  import *
from src.simulation.nb_events   import *

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









@njit
def set_numba_random_seed(seed) :
    np.random.seed(seed)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # INITIAL INFECTIONS  # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@njit
def single_random_choice(x) :
    return np.random.choice(x, size=1)[0]


@njit
def set_to_array(input_set) :
    out = List()
    for s in input_set :
        out.append(s)
    return np.asarray(out)


@njit
def nb_random_choice(arr, prob, size=1, replace=False, verbose=False) :
    """
    :param arr : A 1D numpy array of values to sample from.
    :param prob : A 1D numpy array of probabilities for the given samples.
    :param size : Integer describing the size of the output.
    :return : A random sample from the given array with a given probability.
    """

    assert len(arr) == len(prob)
    assert size < len(arr)

    prob = prob / np.sum(prob)
    if replace :
        ra = np.random.random(size=size)
        idx = np.searchsorted(np.cumsum(prob), ra, side="right")
        return arr[idx]
    else :
        if size / len(arr) > 0.5 and verbose :
            print("Warning : choosing more than 50% of the input array with replacement, can be slow.")

        out = set()
        while len(out) < size :
            ra = np.random.random()
            idx = np.searchsorted(np.cumsum(prob), ra, side="right")
            x = arr[idx]
            if not x in out :
                out.add(x)
        return set_to_array(out)


@njit
def exp_func(x, a, b, c) :
    return a * np.exp(b * x) + c


@njit
def make_random_initial_infections(my, possible_agents, N, prior) :
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
        replace=False,
    )


@njit
def choose_initial_agents(my, possible_agents, N, prior) :

    ##  Standard outbreak type, infecting randomly
    if my.cfg.make_random_initial_infections :
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


@njit
def initialize_states(
    my,
    g,
    SIR_transition_rates,
    state_total_counts,
    infected_per_age_group,
    agents_in_state,
    possible_agents,
    N_init,
    R_init,
    prior_infected,
    prior_immunized,
    verbose=False) :


    R_state = g.N_states - 1

    if R_init > 0 :

        #  Make initial immunizations
        for agent in choose_initial_agents(my, possible_agents, R_init, prior_immunized) :

            # Update the state
            my.state[agent] = R_state

            if np.random.rand() < my.cfg.N_init_UK_frac :
                my.corona_type[agent] = 1

            agents_in_state[R_state].append(np.uint32(agent))

            state_total_counts[R_state] += 1

            g.total_sum_of_state_changes += SIR_transition_rates[R_state]
            g.cumulative_sum_of_state_changes[R_state :] += SIR_transition_rates[R_state]

            # Disable incomming rates
            update_infection_list_for_newly_infected_agent(my, g, agent)


    #  Make initial infections
    for agent in choose_initial_agents(my, possible_agents, N_init, prior_infected) :

        # If infected, do not immunize # TODO: Discuss if this is the best way to immunize agents
        if my.state[agent] == R_state :
            continue

        weights = calc_E_I_distribution(my, 1)
        states = np.arange(g.N_states - 1, dtype=np.int8)
        new_state = nb_random_choice(states, weights, verbose=verbose)[0]  # E1-E4 or I1-I4, uniformly distributed
        my.state[agent] = new_state

        if np.random.rand() < my.cfg.N_init_UK_frac :
            my.corona_type[agent] = 1  # IMPORTANT LINE!

        agents_in_state[new_state].append(np.uint32(agent))
        state_total_counts[new_state] += 1

        g.total_sum_of_state_changes += SIR_transition_rates[new_state]
        g.cumulative_sum_of_state_changes[new_state :] += SIR_transition_rates[new_state]

        # Moves into a infectious State
        if my.agent_is_infectious(agent) :
            for contact, rate in zip(my.connections[agent], g.rates[agent]) :
                # update rates if contact is susceptible
                if my.agent_is_susceptible(contact) :
                    g.update_rates(my, +rate, agent)

            # Update the counters
            infected_per_age_group[my.corona_type[agent]][my.age[agent]] += 1

        # Make sure agent can not be re-infected
        update_infection_list_for_newly_infected_agent(my, g, agent)




    # English Corona Type TODO

    #if my.cfg.N_init_UK > 0 : init uk, as outbreak
    if False :

        rho_init_local_outbreak = 0.1

        possible_agents_UK = np.arange(my.cfg_network.N_tot, dtype=np.uint32)

        # this is where the outbreak starts

        if my.cfg.outbreak_position_UK.lower() == "københavn" :
            coordinate = (55.67594, 12.56553)

            outbreak_agent_UK = find_outbreak_agent(
                my,
                possible_agents_UK,
                coordinate,
                rho_init_local_outbreak,
                max_tries=10_000,
            )

            # print("København", outbreak_agent_UK, my.coordinates[outbreak_agent_UK])

        elif my.cfg.outbreak_position_UK.lower() == "nordjylland" :
            coordinate = (57.36085, 10.09901)  # "Vendsyssel" på Google Maps

            outbreak_agent_UK = find_outbreak_agent(
                my,
                possible_agents_UK,
                coordinate,
                rho_init_local_outbreak,
                max_tries=10_000,
            )
            # print("nordjylland", outbreak_agent_UK, my.coordinates[outbreak_agent_UK])

        # elif "," in my.cfg.outbreak_position_UK :
        # pass
        else :
            outbreak_agent_UK = single_random_choice(possible_agents_UK)
            # print("random", outbreak_agent_UK, my.coordinates[outbreak_agent_UK])

        initial_agents_to_infect_UK = List()
        initial_agents_to_infect_UK.append(outbreak_agent_UK)

        while len(initial_agents_to_infect_UK) < my.cfg.N_init_UK :
            proposed_agent_UK = single_random_choice(possible_agents_UK)

            if my.dist_accepted(outbreak_agent_UK, proposed_agent_UK, rho_init_local_outbreak) :
                if proposed_agent_UK not in initial_agents_to_infect_UK :
                    if my.agent_is_susceptible(proposed_agent_UK) :
                        initial_agents_to_infect_UK.append(proposed_agent_UK)

        initial_agents_to_infect_UK = np.asarray(initial_agents_to_infect_UK, dtype=np.uint32)

        ##  Now make initial UK infections
        for _, agent in enumerate(initial_agents_to_infect_UK) :
            weights = calc_E_I_distribution(my, 1)
            states = np.arange(N_states - 1, dtype=np.int8)
            new_state = nb_random_choice(states, weights)[0]
            my.state[agent] = new_state
            my.corona_type[agent] = 1  # IMPORTANT LINE!

            agents_in_state[new_state].append(np.uint32(agent))
            state_total_counts[new_state] += 1

            g.total_sum_of_state_changes += SIR_transition_rates[new_state]
            g.cumulative_sum_of_state_changes[new_state :] += SIR_transition_rates[new_state]

            # Moves TO infectious State from non-infectious
            if my.agent_is_infectious(agent) :
                for contact, rate in zip(my.connections[agent], g.rates[agent]) :
                    # update rates if contact is susceptible
                    if my.agent_is_susceptible(contact) :
                        g.update_rates(my, +rate, agent)

            update_infection_list_for_newly_infected_agent(my, g, agent)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # PRE SIMULATION  # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


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

    if my.cfg.day_max > 0 and day > my.cfg.day_max :
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

    elif (g.total_sum_infections + g.total_sum_of_state_changes < 0.0001) and (g.total_sum_of_state_changes + g.total_sum_infections > -0.00001) :
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
        print("g.total_sum_infections : \t", g.total_sum_infections)
        print("g.cumulative_sum_infection_rates : \t", g.cumulative_sum_infection_rates)
        print("g.cumulative_sum_of_state_changes : \t", g.cumulative_sum_of_state_changes)
        print("x : \t", x)
        print("ra1 : \t", ra1)
        continue_run = False

    elif (g.total_sum_of_state_changes < 0) and (g.total_sum_of_state_changes > -0.001) :
        g.total_sum_of_state_changes = 0

    elif (g.total_sum_infections < 0) and (g.total_sum_infections > -0.001) :
        g.total_sum_infections = 0

    elif (g.total_sum_of_state_changes < 0) or (g.total_sum_infections < 0) :
        print("\nNegative Problem", g.total_sum_of_state_changes, g.total_sum_infections)
        print("s : \t", s)
        print("g.total_sum_infections : \t", g.total_sum_infections)
        print("g.cumulative_sum_infection_rates : \t", g.cumulative_sum_infection_rates)
        print("g.cumulative_sum_of_state_changes : \t", g.cumulative_sum_of_state_changes)
        print("x : \t", x)
        print("ra1 : \t", ra1)
        continue_run = False

    return continue_run



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
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # SIMULATION  # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@njit
def run_simulation(
    my,
    g,
    intervention,
    SIR_transition_rates,
    state_total_counts,
    infected_per_age_group,
    agents_in_state,
    N_infectious_states,
    nts,
    verbose=False) :

    if verbose :
        print("Apply intervention", intervention.apply_interventions)

    # Define outputs
    out_time = List()                       # Sampled times
    out_state_counts = List()               # Tne counts of the SEIR states
    out_infected_per_age_group = List()     # The counts of infected per age group
    out_my_state = List()

    daily_counter = 0
    day = 0
    click = 0
    step_number = 0

    real_time = 0.0


    s_counter = np.zeros(4)
    where_infections_happened_counter = np.zeros(4)

    start_date_offset = my.cfg.start_date_offset


    # Run the simulation ################################
    continue_run = True
    while continue_run :

        s = 0

        step_number += 1
        g.total_sum = g.total_sum_of_state_changes + g.total_sum_infections

        dt = -np.log(np.random.rand()) / g.total_sum
        real_time += dt

        g.cumulative_sum = 0.0
        ra1 = np.random.rand()

        #######/ Here we move between infected between states
        accept = False
        if g.total_sum_of_state_changes / g.total_sum > ra1 :

            s = 1

            x = g.cumulative_sum_of_state_changes / g.total_sum
            state_now = np.searchsorted(x, ra1)
            state_after = state_now + 1

            agent = utils.numba_random_choice_list(agents_in_state[state_now])

            # We have chosen agent to move -> here we move it
            agents_in_state[state_after].append(agent)
            agents_in_state[state_now].remove(agent)

            my.state[agent] += 1

            state_total_counts[state_now]   -= 1
            state_total_counts[state_after] += 1

            g.total_sum_of_state_changes -= SIR_transition_rates[state_now]
            g.total_sum_of_state_changes += SIR_transition_rates[state_after]

            g.cumulative_sum_of_state_changes[state_now] -= SIR_transition_rates[state_now]
            g.cumulative_sum_of_state_changes[state_after :] += (
                SIR_transition_rates[state_after] - SIR_transition_rates[state_now]
            )

            g.cumulative_sum_infection_rates[state_now] -= g.sum_of_rates[agent]

            accept = True

            if intervention.apply_interventions and intervention.apply_symptom_testing and day >= 0 :
                apply_symptom_testing(my, intervention, agent, click)

            # Moves TO infectious State from non-infectious
            if my.state[agent] == N_infectious_states :
                # for i, (contact, rate) in enumerate(zip(my.connections[agent], g.rates[agent])) :
                for ith_contact, contact in enumerate(my.connections[agent]) :
                    # update rates if contact is susceptible
                    if my.agent_is_connected(agent, ith_contact) and my.agent_is_susceptible(contact) :
                        if my.corona_type[agent] == 1 :
                            g.rates[agent][ith_contact] *= my.cfg.beta_UK_multiplier
                        rate = g.rates[agent][ith_contact]
                        g.update_rates(my, +rate, agent)

                # Update the counters
                infected_per_age_group[my.corona_type[agent]][my.age[agent]] += 1

            # If this moves to Recovered state
            if my.state[agent] == g.N_states - 1 :
                for ith_contact, contact in enumerate(my.connections[agent]) :
                    # update rates if contact is susceptible
                    if my.agent_is_connected(agent, ith_contact) and my.agent_is_susceptible(contact) :
                        rate = g.rates[agent][ith_contact]
                        g.update_rates(my, -rate, agent)

                # Update counters
                infected_per_age_group[my.corona_type[agent]][my.age[agent]] -= 1

        #######/ Here we infect new states
        else :
            s = 2

            x = (g.total_sum_of_state_changes + g.cumulative_sum_infection_rates) / g.total_sum
            state_now = np.searchsorted(x, ra1)
            g.cumulative_sum = (
                g.total_sum_of_state_changes + g.cumulative_sum_infection_rates[state_now - 1]
            ) / g.total_sum  # important change from [state_now] to [state_now-1]

            agent_getting_infected = -1
            for agent in agents_in_state[state_now] :

                # suggested cumulative sum
                suggested_cumulative_sum = g.cumulative_sum + g.sum_of_rates[agent] / g.total_sum

                if suggested_cumulative_sum > ra1 :
                    ith_contact = 0
                    for rate, contact in zip(g.rates[agent], my.connections[agent]) :

                        # if contact is susceptible
                        if my.agent_is_susceptible(contact) :

                            g.cumulative_sum += rate / g.total_sum

                            # here agent infect contact
                            if g.cumulative_sum > ra1 :
                                where_infections_happened_counter[my.connections_type[agent][ith_contact]] += 1
                                my.state[contact] = 0

                                my.corona_type[contact] = my.corona_type[agent]

                                agents_in_state[0].append(np.uint32(contact))
                                state_total_counts[0] += 1
                                g.total_sum_of_state_changes += SIR_transition_rates[0]
                                g.cumulative_sum_of_state_changes += SIR_transition_rates[0]
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

            daily_counter += 1
            if ((len(out_time) == 0) or (real_time != out_time[-1])) and day >= 0 :

                # Update the output variables
                out_time.append(real_time)
                out_state_counts.append(state_total_counts.copy())
                out_infected_per_age_group.append(infected_per_age_group.copy())

            if daily_counter >= 10 :

                # Advance day
                day += 1
                daily_counter = 0

                # Apply interventions
                if intervention.apply_interventions :

                    if intervention.apply_interventions_on_label and day >= 0 :
                        apply_interventions_on_label(my, g, intervention, day, click, verbose)

                    if intervention.apply_random_testing :
                        apply_random_testing(my, intervention, click)

                    if intervention.apply_vaccinations :

                        if start_date_offset > 0 :
                            for day in range(start_date_offset) :
                                vaccinate(my, g, intervention, day, verbose=verbose)

                            intervention.vaccination_schedule + start_date_offset
                            start_date_offset = 0

                        vaccinate(my, g, intervention, day, verbose=verbose)



                # Apply events
                if my.cfg.N_events > 0 :
                    add_daily_events(
                        my,
                        g,
                        day,
                        agents_in_state,
                        state_total_counts,
                        SIR_transition_rates,
                        where_infections_happened_counter)


                if verbose :
                    print("--- day : ", day, " ---")
                    print("n_infected : ", np.round(np.sum(where_infections_happened_counter)))
                    print("R_true : ", np.round(intervention.R_true_list[-1], 3))
                    print("freedom_impact : ", np.round(intervention.freedom_impact_list[-1], 3))
                    print("R_true_list_brit : ", np.round(intervention.R_true_list_brit[-1], 3))


                if day >= 0 :
                    out_my_state.append(my.state.copy())

                    intervention.R_true_list.append(calculate_R_True(my, g))
                    intervention.freedom_impact_list.append(calculate_population_freedom_impact(intervention))
                    intervention.R_true_list_brit.append(calculate_R_True_brit(my, g))

            if intervention.apply_interventions:
                test_tagged_agents(my, g, intervention, day, click)

            click += 1

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
        print("Simulation step_number, ", step_number)
        print("s_counter", s_counter)
        print("Where", where_infections_happened_counter)
        print("positive_test_counter", intervention.positive_test_counter)
        print("n_found", np.sum(np.array([1 for day_found in intervention.day_found_infected if day_found>=0])))
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

    return out_time, out_state_counts, out_infected_per_age_group, out_my_state, intervention
    #return out_time, out_state_counts, out_variant_counts, out_my_state, intervention


# ███    ███  █████  ██████  ████████ ██ ███    ██ ██    ██
# ████  ████ ██   ██ ██   ██    ██    ██ ████   ██  ██  ██
# ██ ████ ██ ███████ ██████     ██    ██ ██ ██  ██   ████
# ██  ██  ██ ██   ██ ██   ██    ██    ██ ██  ██ ██    ██
# ██      ██ ██   ██ ██   ██    ██    ██ ██   ████    ██


@njit
def calculate_contact_distribution(my, contact_type) :
    contact_dist = np.zeros(100)
    for agent in range(my.cfg_network.N_tot) :
        agent_sum = 0
        for ith_contact in range(len(my.connections[agent])) :
            if my.connections_type[agent][ith_contact] == contact_type :
                agent_sum += 1
        contact_dist[agent_sum] += 1
    return contact_dist


@njit
def calculate_contact_distribution_label(my, intervention):
    label_contacts = np.zeros(intervention.N_labels)
    label_people = np.zeros(intervention.N_labels)
    label_infected = np.zeros(intervention.N_labels)
    for agent in range(my.cfg_network.N_tot) :
        label = intervention.labels[agent]

        if my.agent_is_infectious(agent):
            label_infected[label] += 1
        label_people[label] += 1
        label_contacts[label] += len(my.connections[agent])
    return label_contacts, label_infected, label_people


@njit
def vaccinate(my, g, intervention, day, verbose=False) :

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
    intervention,
    day,
    threshold_info) :

    infected_per_label = np.zeros_like(intervention.label_counter, dtype=np.uint32)

    for agent, day_found in enumerate(intervention.day_found_infected) :
        if day_found > max(0, day - intervention.cfg.days_looking_back) :
            infected_per_label[intervention.labels[agent]] += 1


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
    infection_rate = my.infection_weight[agent] * my.beta_connection_type[my.connections_type[agent][ith_contact]]

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

        target_rate = g.rates[contact][ith_contact_of_contact] * rate_multiplication[my.connections_type[contact][ith_contact_of_contact]]

        # Update the g.rates
        rate = target_rate - g.rates[contact][ith_contact_of_contact]
        g.rates[contact][ith_contact_of_contact] = target_rate

        # Updates to gillespie sums
        if my.agent_is_infectious(contact) and my.agent_is_susceptible(agent):
            g.update_rates(my, rate, contact)


@njit
def remove_intervention_at_label(my, g, intervention, ith_label) :
    for agent in range(my.cfg_network.N_tot) :
        if intervention.labels[agent] == ith_label and my.restricted_status[agent] == 1 : #TODO: Only if not tested positive
            reset_rates_of_agent(my, g, agent, intervention)
            my.restricted_status[agent] = 0


@njit
def check_if_intervention_on_labels_can_be_removed(my, g, intervention, day, click,  threshold_info) :

    infected_per_label = np.zeros(intervention.N_labels, dtype=np.int32)
    for agent, day_found in enumerate(intervention.day_found_infected) :
        if day_found > day - intervention.cfg.days_looking_back :
            infected_per_label[intervention.labels[agent]] += 1

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
                * rate_reduction[my.connections_type[contact][ith_contact_of_contact]]
            )
            intervention.freedom_impact[contact] += rate_reduction[my.connections_type[contact][ith_contact_of_contact]]/my.number_of_contacts[contact]
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

        rate = g.rates[agent][ith_contact] * rate_reduction[my.connections_type[agent][ith_contact]]
        intervention.freedom_impact[contact] += rate_reduction[my.connections_type[agent][ith_contact]]/my.number_of_contacts[agent]

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
        if np.random.rand() > remove_rates[my.connections_type[agent][ith_contact]] :
            act_rate_reduction = np.array([0, 0, 0], dtype=np.float64)
        else :
            act_rate_reduction = reduce_rates

        rate = (
            g.rates[agent][ith_contact]
            * act_rate_reduction[my.connections_type[agent][ith_contact]]
        )
        intervention.freedom_impact[agent] += act_rate_reduction[my.connections_type[agent][ith_contact]]/my.number_of_contacts[agent]
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
        if np.random.rand() < remove_rates[my.connections_type[agent][ith_contact]] :
            act_rate_reduction = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        rate = (
            g.rates[agent][ith_contact]
            * act_rate_reduction[my.connections_type[agent][ith_contact]]
        )

        g.rates[agent][ith_contact] -= rate
        intervention.freedom_impact[agent] += act_rate_reduction[my.connections_type[agent][ith_contact]]/my.number_of_contacts[agent]

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
        work_matrix_previous  = my.cfg_network.work_matrix[label]
        other_matrix_previous = my.cfg_network.other_matrix[label]
    else :
        work_matrix_previous  = intervention.work_matrix_restrict[label][n-1]
        other_matrix_previous = intervention.other_matrix_restrict[label][n-1]

    work_matrix_max  = my.cfg_network.work_matrix
    other_matrix_max = my.cfg_network.other_matrix

    work_matrix_current  = intervention.work_matrix_restrict[label][n]
    other_matrix_current = intervention.other_matrix_restrict[label][n]


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
        if not my.connections_type[agent][ith_contact] == 0 :

            if my.connections_type[agent][ith_contact] == 1 :
                sc = work_matrix_current[my.age[agent], my.age[contact]]
                sp = work_matrix_previous[my.age[agent], my.age[contact]]
                sm = work_matrix_max[my.age[agent], my.age[contact]]

            elif my.connections_type[agent][ith_contact] == 2 :
                sc = other_matrix_current[my.age[agent], my.age[contact]]
                sp = other_matrix_previous[my.age[agent], my.age[contact]]
                sm = other_matrix_max[my.age[agent], my.age[contact]]

            connection_probability_current[ith_contact]  = sc / sm
            connection_probability_previous[ith_contact] = sp / sm

    # Step 2, check if the changes should close any of the active connections
    # Store the connection weight
    sum_connection_probability = np.sum(connection_probability_current)

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
            p = 1 - np.sqrt(1 - connection_probability_current[ith_contact])
            if np.random.rand() < p :
                open_connection(my, g, agent, ith_contact, intervention)

@njit
def lockdown_on_label(my, g, intervention, label, rate_reduction) :
    # lockdown on all agent with a certain label (tent or municipality, or whatever else you define). Rate reduction is two vectors of length 3. First is the fraction of [home, job, others] rates to set to 0.
    # second is the fraction of reduction of the remaining [home, job, others] rates.
    # ie : [[0,0.8,0.8],[0,0.8,0.8]] means that 80% of your contacts on job and other is set to 0, and the remaining 20% is reduced by 80%.
    # loop over all agents
    for agent in range(my.cfg_network.N_tot) :
        if intervention.labels[agent] == label :
            my.restricted_status[agent] = 1
            remove_and_reduce_rates_of_agent(my, g, intervention, agent, rate_reduction)


@njit
def masking_on_label(my, g, intervention, label, rate_reduction) :
    # masking on all agent with a certain label (tent or municipality, or whatever else you define). Rate reduction is two vectors of length 3. First is the fraction of [home, job, others] rates to be effected by masks.
    # second is the fraction of reduction of the those [home, job, others] rates.
    # ie : [[0,0.2,0.2],[0,0.8,0.8]] means that your wear mask when around 20% of job and other contacts, and your rates to those is reduced by 80%
    # loop over all agents
    for agent in range(my.cfg_network.N_tot) :
        if intervention.labels[agent] == label :
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
        print("Ratio")
        print(after / prev)


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
                    < intervention.cfg.tracing_rates[my.connections_type[agent][ith_contact]]
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
def apply_symptom_testing(my, intervention, agent, click) :

    if my.agent_is_infectious(agent) :

        prob = intervention.cfg.chance_of_finding_infected[my.state[agent] - 4] # TODO: Fjern hardcoded 4
        randomly_selected = np.random.rand() < prob
        not_tested_before = (intervention.clicks_when_tested[agent] == -1)

        if randomly_selected and not_tested_before :
            # testing in n_clicks for symptom checking
            intervention.clicks_when_tested[agent] = (
                click + intervention.cfg.test_delay_in_clicks[0]
            )
            # set the reason for testing to symptoms (0)
            intervention.reason_for_test[agent] = 0


@njit
def apply_random_testing(my, intervention, click) :
    # choose N_daily_test people at random to test
    N_daily_test = int(my.cfg.daily_tests * my.cfg_network.N_tot / 5_800_000)
    agents = np.arange(my.cfg_network.N_tot, dtype=np.uint32)
    random_agents_to_be_tested = np.random.choice(agents, N_daily_test)
    intervention.clicks_when_tested[random_agents_to_be_tested] = (
        click + intervention.cfg.test_delay_in_clicks[1]
    )
    # specify that random test is the reason for test
    intervention.reason_for_test[random_agents_to_be_tested] = 1


@njit
def apply_interventions_on_label(my, g, intervention, day, click, verbose=False) :

    if intervention.start_interventions_by_real_incidens_rate or intervention.start_interventions_by_meassured_incidens_rate :
        threshold_info = np.array([[1, 2], [200, 100], [20, 20]]) #TODO : remove
        check_if_intervention_on_labels_can_be_removed(my, g, intervention, day, click, threshold_info)

        for i_label, clicks_when_restriction_stops in enumerate(intervention.clicks_when_restriction_stops) :
            if clicks_when_restriction_stops == click :
                remove_intervention_at_label(my, g, intervention, i_label)
                intervention.clicks_when_restriction_stops[i_label] = -1
                intervention_type_n = intervention.types[i_label]
                intervention.types[i_label] = 0
                intervention.started_as[i_label] = 0
                if intervention.verbose :
                    intervention_type_name = ["nothing", "lockdown", "masking", "error", "error", "error", "error", "matrix_based"]
                    print(
                        *("remove ", intervention_type_name[intervention_type_n], " at num of infected", i_label),
                        *("at day", day)
                    )

        check_if_label_needs_intervention(
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
                                    print("Intervention type : lockdown")

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
                                    print("Intervention type : masks")

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
                                    print("Intervention type : matrix restriction, name: ", intervention.cfg.Intervention_contact_matrices_name[int(i/2)])

                                matrix_restriction_on_label(
                                    my,
                                    g,
                                    intervention,
                                    ith_label,
                                    int(i/2),
                                    verbose=verbose
                                )
                    else :
                        for i_label, intervention_type in enumerate(intervention.types) :

                            # Matrix restrictions are not removed # TODO: Remove if no restrictions follow
                            if intervention.cfg.threshold_interventions_to_apply[int(i/2)] == 3 :
                                continue

                            if verbose :
                                print("Intervention removed")

                            remove_intervention_at_label(my, g, intervention, i_label)





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
                rate_reduction=intervention.cfg.isolation_rate_reduction,
            )

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
                    rate_reduction=intervention.cfg.isolation_rate_reduction,
                )

