import numpy as np

from numba import njit

from numba.typed import List

from src.utils import utils

from src.simulation.nb_interventions   import test_tagged_agents, vaccinate, apply_symptom_testing, apply_daily_interventions
from src.simulation.nb_interventions   import calculate_R_True, calculate_R_True_brit, calculate_population_freedom_impact
from src.simulation.nb_events          import add_daily_events
from src.simulation.nb_initialize      import update_infection_list_for_newly_infected_agent

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


@njit
def run_simulation(
    my,
    g,
    intervention,
    state_total_counts,
    stratified_infection_counts,
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
            for d in range(-my.cfg.start_date_offset, 0) :
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
                stratified_infection_counts[my.label[agent]][my.corona_type[agent]][my.age[agent]] += 1

            # If this moves to Recovered state
            if my.state[agent] == g.N_states - 1 :
                for ith_contact, contact in enumerate(my.connections[agent]) :
                    # update rates if contact is susceptible
                    if my.agent_is_connected(agent, ith_contact) and my.agent_is_susceptible(contact) :
                        rate = g.rates[agent][ith_contact]
                        g.update_rates(my, -rate, agent)

                # Update counters
                stratified_infection_counts[my.label[agent]][my.corona_type[agent]][my.age[agent]] -= 1

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

            # Advance click
            click += 1
            daily_counter += 1

            # Apply interventions on clicks
            if intervention.apply_interventions:
                test_tagged_agents(my, g, intervention, day, click)

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
                    print("--- day : ", day, " ---")
                    print("n_infected : ", np.round(my.cfg.N_init + np.sum(where_infections_happened_counter)))
                    print("R_true : ", np.round(intervention.R_true_list[-1], 3))
                    print("freedom_impact : ", np.round(intervention.freedom_impact_list[-1], 3))
                    print("R_true_list_brit : ", np.round(intervention.R_true_list_brit[-1], 3))


                # Advance day
                day += 1
                daily_counter = 0

                # Apply interventions for the new day
                if intervention.apply_interventions :
                    apply_daily_interventions(my, g, intervention, day, click, stratified_vaccination_counts, verbose)

                # Apply events for the new day
                if my.cfg.N_events > 0 :
                    add_daily_events(
                        my,
                        g,
                        day,
                        agents_in_state,
                        state_total_counts,
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

    return out_time, out_state_counts, out_stratified_infection_counts, out_stratified_vaccination_counts, out_my_state, intervention
    #return out_time, out_state_counts, out_variant_counts, out_my_state, intervention
